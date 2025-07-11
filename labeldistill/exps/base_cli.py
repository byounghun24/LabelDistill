# Copyright (c) Megvii Inc. All rights reserved.
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from labeldistill.callbacks.ema import EMACallback
from labeldistill.utils.torch_dist import all_gather_object, synchronize

from .nuscenes.base_exp import LabelDistillModel
from datetime import datetime

def run_cli(model_class=LabelDistillModel,
            exp_name='base_exp',
            use_ema=False,
            extra_trainer_config_args={}):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = LabelDistillModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=extra_trainer_config_args.get('epochs', 24),
                        strategy='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=0.00,
                        enable_checkpointing=True,
                        precision=16,
                        default_root_dir=os.path.join('./outputs/', exp_name),
                        log_every_n_steps=1)
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    wandb_logger = WandbLogger(
        name='CL_debug_' + exp_name + '_' + current_time,
        project="LabelDistill",
        save_dir=os.path.join('./outputs/', exp_name),
        log_model=True
    )

    model = model_class(**vars(args))

    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[ema_callback])
    else:
        trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)

    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)
