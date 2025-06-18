from torch import nn

from labeldistill.layers.backbones.base_lss_fpn import BaseLSSFPN
from labeldistill.layers.backbones.adaptor import DistillAdaptor
from labeldistill.layers.backbones.label_backbone import LabelBackbone

from labeldistill.layers.heads.kd_head import KDHead
from labeldistill.layers.heads.kd_fpn import KDFPN
from mmdet3d.models import build_detector
import torch

__all__ = ['LabelDistill']

"""
Label and LiDAR Distillation (backbone to backbone)
"""
class LabelDistill(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self,
                 backbone_conf,
                 head_conf,
                 labelenc_conf,
                 lidar_conf=None,
                 le_ckpt_path=None,
                 lidar_ckpt_path=None,
                 is_train_depth=False,
                 ):
        super(LabelDistill, self).__init__()
        self.backbone = BaseLSSFPN(**backbone_conf)
        self.kdfpn = KDFPN(**head_conf)
        self.fusion = SimpleFusion(150, 100)
        self.head = KDHead(**head_conf)
        self.is_train_depth = is_train_depth

        distill_in_feature = head_conf['bev_neck_conf']['in_channels'][:2]
        self.distill_encoder_lidar = DistillAdaptor([x // 3 for x in distill_in_feature],
                                                    out_features=[128, 256],
                                                    stride=[1, 1]
                                                    )
        self.distill_encoder_label = DistillAdaptor([x // 3 for x in distill_in_feature],
                                                    out_features=[128, 256],
                                                    stride=[1, 1]
                                                    )

        # build lidar detection model
        self.centerpoint = build_detector(lidar_conf)

        # load pretrained parameters for lidar detection model
        lidar_params = torch.load(lidar_ckpt_path, map_location='cpu')

        prefix = 'model.centerpoint.'
        load_keys = [k for k in lidar_params['state_dict'] if k.startswith(prefix)]
        self.centerpoint.load_state_dict({k[len(prefix):]: lidar_params['state_dict'][k] for k in load_keys})
        self.centerpoint.eval()

        # build label encoder model
        self.label_encoder = LabelBackbone(**labelenc_conf)

        # load pretrained parameters for label encoder model
        le_ckpt = torch.load(le_ckpt_path, map_location='cpu')
        prefix = 'model.label_encoder.'
        load_keys = [k for k in le_ckpt['state_dict'] if k.startswith(prefix)]
        self.label_encoder.load_state_dict({k[len(prefix):]: le_ckpt['state_dict'][k] for k in load_keys}, strict=False)
        self.label_encoder.eval()


    def forward(
        self,
        bev_mask=None,
        bev_box=None,
        bev_label=None,
        x=None,
        mats_dict=None,
        lidar_pts=None,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """

        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True) # x.shape = [B,C,W,H]

            with torch.no_grad():
                # label feature generation
                label_feats = self.label_encoder(bev_box, bev_label, bev_mask.unsqueeze(1))

                # lidar feature generation
                voxels, num_points, coors = self.centerpoint.voxelize(lidar_pts.squeeze(1))
                voxel_features = self.centerpoint.pts_voxel_encoder(voxels, num_points, coors)
                batch_size = coors[-1, 0] + 1
                lidar_feats = self.centerpoint.pts_middle_encoder(voxel_features, coors, batch_size)
                lidar_feats = self.centerpoint.pts_backbone(lidar_feats)
                lidar_feats_out = lidar_feats

                # lidar detection output generation
                lidar_feats = self.centerpoint.pts_neck(lidar_feats)
                lidar_preds = self.centerpoint.pts_bbox_head(lidar_feats)

            after_fpn_all = self.kdfpn(x)
            after_fpn = after_fpn_all[:2]

            c1 = after_fpn[0].shape[1] // 3
            c2 = after_fpn[1].shape[1] // 3

            after_fusion_large, after_fusion_small, label_feats_large, label_feats_small, lidar_feats_large, lidar_feats_small = self.fusion(after_fpn, c1, c2)

            after_fusion = [after_fusion_large, after_fusion_small, after_fpn_all[2], after_fpn_all[3]]

            #adaptor
            distill_feats_label = self.distill_encoder_label([label_feats_large,
                                                              label_feats_small])
            distill_feats_lidar = self.distill_encoder_lidar([lidar_feats_large,
                                                              lidar_feats_small])
            
            preds = self.head(after_fusion) #backbone output list of [B, C, W, H]

            return preds, lidar_preds, depth_pred, distill_feats_lidar, lidar_feats_out, distill_feats_label, label_feats
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds, _ = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def response_loss(self, targets, preds_dicts, teacher_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.response_loss(targets, preds_dicts, teacher_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)


class SimpleFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        def make_fusion_block(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )

        self.fuse_large = make_fusion_block(c1 * 3)
        self.fuse_small = make_fusion_block(c2 * 3)

    def forward(self, after_fpn, c1, c2):
        # Split modalities
        label_feats_large = after_fpn[0][:, :c1]
        label_feats_small = after_fpn[1][:, :c2]

        lidar_feats_large = after_fpn[0][:, c1:c1*2]
        lidar_feats_small = after_fpn[1][:, c2:c2*2]

        camera_feats_large = after_fpn[0][:, c1*2:]
        camera_feats_small = after_fpn[1][:, c2*2:]

        # Concatenate: [label | lidar | camera] along channel dim
        large_concat = torch.cat([label_feats_large, lidar_feats_large, camera_feats_large], dim=1)
        small_concat = torch.cat([label_feats_small, lidar_feats_small, camera_feats_small], dim=1)

        # Fusion
        fused_large = self.fuse_large(large_concat)
        fused_small = self.fuse_small(small_concat)

        return fused_large, fused_small, label_feats_large, label_feats_small, lidar_feats_large, lidar_feats_small