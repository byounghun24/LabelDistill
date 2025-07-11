# Copyright (c) Megvii Inc. All rights reserved.
"""
AP: 0.3929
mATE: 0.6216
mASE: 0.2571
mAOE: 0.4407
mAVE: 0.3622
mAAE: 0.2099
NDS: 0.5073
Eval time: 65.0s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.592   0.444   0.153   0.087   0.367   0.199
truck                   0.352   0.593   0.193   0.103   0.317   0.184
bus                     0.418   0.610   0.172   0.065   0.782   0.263
trailer                 0.211   0.881   0.219   0.441   0.248   0.191
construction_vehicle    0.102   1.053   0.476   1.194   0.126   0.411
pedestrian              0.360   0.703   0.281   0.740   0.404   0.226
motorcycle              0.396   0.602   0.245   0.526   0.491   0.200
bicycle                 0.367   0.473   0.251   0.687   0.161   0.006
traffic_cone            0.552   0.447   0.318   nan     nan     nan
barrier                 0.579   0.410   0.262   0.125   nan     nan
"""

from labeldistill.exps.base_cli import run_cli
from labeldistill.exps.nuscenes.base_exp import \
    LabelDistillModel as BaseLabelDistillModel
from labeldistill.models.labeldistill_CL import LabelDistill
from torch.optim.lr_scheduler import MultiStepLR
from mmcv.runner import build_optimizer
from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from functools import partial

import torch
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from torchvision.ops import roi_align
import torch.nn as nn
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d

def plot_anchor_positive_similarity(sim_matrix, anchor_labels, positive_labels, max_points=64):
    Na = min(len(anchor_labels), max_points)
    Np = min(len(positive_labels), max_points)
    sim_matrix = sim_matrix[:Na, :Np]
    anchor_labels = anchor_labels[:Na]
    positive_labels = positive_labels[:Np]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sim_matrix, xticklabels=positive_labels, yticklabels=anchor_labels,
                cmap='coolwarm', square=True, cbar=True, ax=ax)
    ax.set_title("Anchor vs Positive Similarity")
    ax.set_xlabel("Positive Features (class)")
    ax.set_ylabel("Anchor Features (class)")
    plt.tight_layout()
    return fig

class LabelDistillModel(BaseLabelDistillModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.key_idxes = [-2, -4]

        self.backbone_conf['output_channels'] = 150
        self.head_conf['bev_backbone_conf']['in_channels'] = 150 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_backbone_conf']['base_channels'] = 150 * 2
        self.head_conf['bev_neck_conf']['in_channels'] = [
            150 * (len(self.key_idxes) + 1), 150*2, 150*4, 150*8
        ]
        self.head_conf['train_cfg']['code_weights'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]

        self.data_return_lidar = True

        self.optimizer_config = dict(
            type='AdamW',
            lr=4e-4,
            paramwise_cfg=dict(
                custom_keys={
                    'backbone': dict(lr_mult=0.5),
                }),
            weight_decay=1e-2)

        #############################################################################################
        """
        Models:
          - Name: centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d
            In Collection: CenterPoint
            Config: configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py
            metadata:
              Training Memory (GB): 5.2
            Results:
              - Task: 3D Object Detection
                Dataset: nuScenes
                Metrics:
                  mAP: 56.11
                  NDS: 64.61
            Weights: https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth
        """

        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size = [0.1, 0.1, 0.2]

        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            code_size=9)

        train_cfg = dict(
            pts=dict(
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
        test_cfg = dict(
            pts=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                nms_type='circle',
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2))

        self.lidar_conf = dict(type='CenterPoint',
            pts_voxel_layer=dict(
                point_cloud_range=point_cloud_range, max_num_points=10, voxel_size=voxel_size,
                max_voxels=(90000, 120000)),
            pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
            pts_middle_encoder=dict(
                type='SparseEncoder',
                in_channels=5,
                sparse_shape=[41, 1024, 1024],
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                block_type='basicblock'),
            pts_backbone=dict(
                type='SECOND',
                in_channels=256,
                out_channels=[128, 256],
                layer_nums=[5, 5],
                layer_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                conv_cfg=dict(type='Conv2d', bias=False)),
            pts_neck=dict(
                type='SECONDFPN',
                in_channels=[128, 256],
                out_channels=[256, 256],
                upsample_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True),
             pts_bbox_head=dict(
                 type='CenterHead',
                 in_channels=sum([256, 256]),
                 tasks=[
                     dict(num_class=1, class_names=['car']),
                     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                     dict(num_class=2, class_names=['bus', 'trailer']),
                     dict(num_class=1, class_names=['barrier']),
                     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 common_heads=dict(
                     reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
                 share_conv_channel=64,
                 bbox_coder=bbox_coder,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                 norm_bbox=True),
        )
        #############################################################################################
        "reproduced centerpoint"
        lidar_ckpt_path = './ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth'

        "original centerpoint"
        # lidar_ckpt_path = './pretrained/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth'
        #############################################################################################
        self.labelenc_conf = dict(
            box_features=9,
            label_features=10,
            hidden_features=256,
            out_features=[128, 256],
            stride=[1, 2],
            feature_size=128
        )

        le_ckpt_path='./ckpts/label_encoder_pretrained.pth'
        #############################################################################################

        self.model = LabelDistill(self.backbone_conf,
                                    self.head_conf,
                                    self.labelenc_conf,
                                    self.lidar_conf,
                                    le_ckpt_path,
                                    lidar_ckpt_path,
                                    is_train_depth=True)
        ##############################################################################################
        proj_dims = self.labelenc_conf['out_features']  # 예: [128, 256]
        self.proj_input_size = 1
        self.cam_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C * self.proj_input_size * self.proj_input_size, C),
                nn.LayerNorm(C),
                nn.ReLU(inplace=True),
                nn.Linear(C, C)
            ) for C in proj_dims
        ])
        self.lidar_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C * self.proj_input_size * self.proj_input_size, C),
                nn.LayerNorm(C),
                nn.ReLU(inplace=True),
                nn.Linear(C, C)
            ) for C in proj_dims
        ])

        # 가중치 초기화 (He init) for Linear layers
        for proj_list in (self.cam_proj_heads, self.lidar_proj_heads):
            for m in proj_list:
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        nn.init.zeros_(layer.bias)
        self.pc_range = point_cloud_range
        ################################################################################################

    def training_step(self, batch):
        (sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels, lidar_pts, depth_labels) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes] # [N,9]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels] # [N]
            self.model = self.model.cuda()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            bev_mask, bev_box, bev_label, targets = self.model.module.get_targets(gt_boxes, gt_labels)
        else:
            bev_mask, bev_box, bev_label, targets = self.model.get_targets(gt_boxes, gt_labels) # [B,128,128], [B,128,128,9], [B,128,128,10], []

        preds, lidar_preds, depth_preds, distill_feats_lidar, lidar_feats, distill_feats_label, label_feats = self.model(bev_mask,
                                                                                                                         bev_box,
                                                                                                                         bev_label,
                                                                                                                         sweep_imgs,
                                                                                                                         mats,
                                                                                                                         lidar_pts)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            detection_loss, response_loss = self.model.module.response_loss(targets, preds, lidar_preds)
        else:
            detection_loss, response_loss = self.model.response_loss(targets, preds, lidar_preds)

        results, lidar_results = self.compute_contrastive_pairs(preds, lidar_preds, img_metas, gt_boxes)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        lidar_distill_loss = self.get_feature_distill_loss(lidar_feats, distill_feats_lidar, targets[0], binary_mask=False) * 0.3
        label_distill_loss = self.get_feature_distill_loss(label_feats, distill_feats_label, targets[0], binary_mask=True) * 0.3

        contrastive_loss = self.compute_contrastive_loss(
            bev_box, bev_label, distill_feats_lidar, lidar_feats,
            results, lidar_results
        ) * 200

        self.log('detection_loss', detection_loss)
        self.log('response_loss', response_loss)
        self.log('depth_loss', depth_loss)
        self.log('lidar_distill_loss', lidar_distill_loss)
        self.log('label_distill_loss', label_distill_loss)
        self.log('contrastive_loss', contrastive_loss)

        return detection_loss + depth_loss + label_distill_loss + response_loss + contrastive_loss

    def compute_contrastive_pairs(self, preds, lidar_preds, img_metas, gt_boxes_list, iou_thresh=0.5, score_thresh=0.3):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)              # camera
            lidar_results = self.model.module.get_bboxes(lidar_preds, img_metas)  # lidar
        else:
            results = self.model.get_bboxes(preds, img_metas)
            lidar_results = self.model.get_bboxes(lidar_preds, img_metas)

        camera_filtered_list = []
        lidar_filtered_list = []

        for b, (cam_res, lidar_res, gt_boxes) in enumerate(zip(results, lidar_results, gt_boxes_list)):
            # camera
            cam_boxes, cam_scores, _ = cam_res  # label은 폐기
            # lidar
            lidar_boxes_full, lidar_scores, lidar_labels = lidar_res

            # (1) Lidar filtering: GT와 IoU > threshold
            lidar_boxes_iou = lidar_boxes_full[:, :7]
            gt_boxes_iou = gt_boxes[:, :7].to(lidar_boxes_iou.device)

            ious = bbox_overlaps_3d(lidar_boxes_iou, gt_boxes_iou)  # (N_lidar, N_gt)
            max_ious, _ = ious.max(dim=1)
            valid_mask = max_ious > iou_thresh

            lidar_boxes_filtered = lidar_boxes_full[valid_mask]
            lidar_scores_filtered = lidar_scores[valid_mask]
            lidar_labels_filtered = lidar_labels[valid_mask]

            num_valid = lidar_boxes_filtered.shape[0]

            # (2) Camera filtering: match with filtered lidar using IoU + confidence
            if num_valid > 0 and cam_boxes.shape[0] > 0:
                lidar_boxes_for_iou = lidar_boxes_filtered[:, :7]
                cam_boxes_for_iou = cam_boxes[:, :7]

                ious_lidar_to_cam = bbox_overlaps_3d(lidar_boxes_for_iou, cam_boxes_for_iou)  # (M, N)
                matched_cam_idxs = ious_lidar_to_cam.argmax(dim=1)                            # (M,)
                max_ious_per_lidar = ious_lidar_to_cam.max(dim=1).values
                cam_scores_matched = cam_scores[matched_cam_idxs]

                valid_score_mask = (max_ious_per_lidar > 0) & (cam_scores_matched > score_thresh)

                final_lidar_idxs = valid_score_mask.nonzero(as_tuple=True)[0]
                final_cam_idxs = matched_cam_idxs[valid_score_mask]

                if final_cam_idxs.numel() > 0:
                    # 중복 camera box 제거 & 매칭된 lidar label 선택
                    unique_cam_idxs, inverse_map = torch.unique(final_cam_idxs, return_inverse=True)
                    candidate_labels = lidar_labels_filtered[final_lidar_idxs]

                    selected_labels = []
                    for u in range(len(unique_cam_idxs)):
                        first_match_idx = (inverse_map == u).nonzero(as_tuple=True)[0][0]
                        selected_labels.append(candidate_labels[first_match_idx].item())

                    cam_boxes_filtered = cam_boxes[unique_cam_idxs]
                    cam_scores_filtered = cam_scores[unique_cam_idxs]
                    cam_labels_filtered = torch.tensor(selected_labels, dtype=torch.long, device=cam_boxes.device)
                else:
                    cam_boxes_filtered = cam_boxes[:0]
                    cam_scores_filtered = cam_scores[:0]
                    cam_labels_filtered = cam_scores[:0].long()
            else:
                cam_boxes_filtered = cam_boxes[:0]
                cam_scores_filtered = cam_scores[:0]
                cam_labels_filtered = cam_scores[:0].long()

            # Append per batch
            lidar_filtered_list.append((lidar_boxes_filtered, lidar_scores_filtered, lidar_labels_filtered))
            camera_filtered_list.append((cam_boxes_filtered, cam_scores_filtered, cam_labels_filtered))

        return camera_filtered_list, lidar_filtered_list

    def compute_contrastive_loss(
        self,
        bev_box,                # (B, H, W, 9)
        bev_label,              # (B, H, W, C)
        cam_feats_list,         # list[(B, C_i, H, W)]
        lidar_feats_list,       # list[(B, C_i, H, W)]
        results,                # camera_filtered_list ([(boxes, scores, labels), ...])
        lidar_results,          # lidar_filtered_list ([(boxes, scores, labels), ...])
        temperature=0.07,
        eps=1e-8
    ):
        num_classes = bev_label.shape[-1]
        total_loss = 0.0
        valid_batches = 0

        for level, (cam_feats, lidar_feats) in enumerate(zip(cam_feats_list, lidar_feats_list)):
            B, C, H, W = cam_feats.shape
            cam_proj = self.cam_proj_heads[level]
            lidar_proj = self.lidar_proj_heads[level]

            anchors, anchor_labels = [], []
            positives, positive_labels = [], []

            for b in range(B):
                cam_boxes, _, cam_labels = results[b]
                lidar_boxes, _, lidar_labels = lidar_results[b]

                # 배치 인덱스가 같이 들어가야 하므로
                def box_to_roi(boxes):
                    rois = []
                    for box in boxes:
                        cx, cy, _, dx, dy, _, *_ = box.tolist()
                        x1 = (cx - dx / 2 - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]) * W
                        y1 = (cy - dy / 2 - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]) * H
                        x2 = (cx + dx / 2 - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]) * W
                        y2 = (cy + dy / 2 - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]) * H
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W - 1, x2), min(H - 1, y2)
                        rois.append([0, x1, y1, x2, y2])
                    return torch.tensor(rois, dtype=torch.float, device=cam_feats.device)

                if cam_boxes.shape[0] > 0 and lidar_boxes.shape[0] > 0:
                    cam_rois = box_to_roi(cam_boxes)
                    lidar_rois = box_to_roi(lidar_boxes)

                    cam_roi_feats = roi_align(cam_feats[b:b+1], cam_rois, output_size=(self.proj_input_size, self.proj_input_size), aligned=True)
                    lidar_roi_feats = roi_align(lidar_feats[b:b+1], lidar_rois, output_size=(self.proj_input_size, self.proj_input_size), aligned=True)

                    cam_flat = F.normalize(cam_roi_feats.view(cam_roi_feats.size(0), -1), dim=1)
                    lidar_flat = F.normalize(lidar_roi_feats.view(lidar_roi_feats.size(0), -1), dim=1)

                    cam_vecs = cam_proj(cam_flat)
                    lidar_vecs = lidar_proj(lidar_flat)

                    anchors.append(cam_vecs)
                    anchor_labels.append(cam_labels)
                    positives.append(lidar_vecs)
                    positive_labels.append(lidar_labels)

            if not anchors:
                continue

            anchors = torch.cat(anchors, dim=0)
            positives = torch.cat(positives, dim=0)
            anchor_labels = torch.cat(anchor_labels, dim=0)
            positive_labels = torch.cat(positive_labels, dim=0)

            # normalize for cosine similarity
            anchors = F.normalize(anchors, dim=1)
            positives = F.normalize(positives, dim=1)

            cos_sim = anchors @ positives.T  # (Na, Np)

            ################# Visualize cosine similarity matrix ###################
            classes_vis = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            cos_sim_np = cos_sim.detach().cpu().numpy()

            anchor_labels_np = anchor_labels.detach().cpu().numpy()  # (Na,)
            positive_labels_np = positive_labels.detach().cpu().numpy()  # (Np,)

            anchor_ids = [f"{classes_vis[cls]}:{i}" for i, cls in enumerate(anchor_labels_np)]
            positive_ids = [f"{classes_vis[cls]}:{i}" for i, cls in enumerate(positive_labels_np)]

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cos_sim_np, xticklabels=positive_ids, yticklabels=anchor_ids,
                        cmap="viridis", vmin=-1, vmax=1, ax=ax)
            ax.set_xlabel("LiDAR_feat (class:idx)")
            ax.set_ylabel("Camera_feat (class:idx)")
            ax.set_title("Camera_feat vs LiDAR_feat Cosine Similarity")

            wandb.log({"cosine_similarity_heatmap": wandb.Image(fig)})
            plt.close(fig)
            ##########################################################################

            cos_sim = cos_sim / temperature
            logits = cos_sim - cos_sim.max(dim=1, keepdim=True).values.detach()

            # same class label → positive
            mask = (anchor_labels.unsqueeze(1) == positive_labels.unsqueeze(0)).float()

            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)
            mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + eps)
            loss = -mean_log_prob_pos.mean()

            total_loss += loss
            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, requires_grad=True, device=bev_label.device)
        return total_loss / valid_batches

    def get_feature_distill_loss(self, lidar_feat, distill_feats, bev_mask=None, binary_mask=False):

        label_losses = 0

        if bev_mask is not None:
            bev_mask = torch.cat(bev_mask, dim=1).sum(1).unsqueeze(1)
            B, _, W, H = bev_mask.shape

            bev_mask = [bev_mask,
                        F.interpolate(bev_mask.type(torch.float32), size=(W//2, H//2), mode='bilinear', align_corners=True)]

            if binary_mask:
                bev_mask[0][bev_mask[0] > 0] = 1.0
                bev_mask[1][bev_mask[1] > 0] = 1.0

        for i in range(len(lidar_feat)):
            label_loss = F.mse_loss(
                lidar_feat[i],
                distill_feats[i],
                reduction='none',
            )

            if bev_mask is not None:
                label_loss = ((label_loss.sum(1) * bev_mask[i].squeeze()).sum()) / max(1.0, bev_mask[i].sum())
            else:
                B, C, W, H = label_loss.shape
                label_loss = label_loss.sum() / (B*W*H)
            label_losses += label_loss

        return label_losses

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(x=sweep_imgs, mats_dict=mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def train_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=self.train_info_paths,
                                       is_train=True,
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=self.data_return_depth,
                                       return_lidar=self.data_return_lidar,
                                       use_fusion=self.use_fusion)

        print(train_dataset.classes)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion,
                               is_return_lidar=self.data_return_lidar),
            sampler=None,
        )
        return train_loader

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.optimizer_config)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(LabelDistillModel,
            'LiDARandLabelDistill_r50_128x128_e24_2key',
            extra_trainer_config_args={'epochs': 24},
            use_ema=True)