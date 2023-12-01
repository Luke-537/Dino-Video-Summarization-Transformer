import argparse
import json
import csv
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.io as io
from pathlib import Path
from torch import nn
from tqdm import tqdm
from torchvision.transforms import functional as tf

from datasets import UCF101, HMDB51, Kinetics, DinoLossLoader, FrameSelectionLoader
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
from visualization import save_tensor_as_video
from models.timesformer import TimeSformer


def timesformer_classification(args):

    config = load_config(args)
    model = get_vit_base_patch16_224(cfg=config, no_head=True)
    ckpt = torch.load(args.pretrained_weights)
    renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
    model.load_state_dict(renamed_checkpoint, strict=False)
    model.eval()

    dataset_test = FrameSelectionLoader(
        cfg=config,
        mode="test",
        loss_file="/home/reutemann/Dino-Video-Summarization-Transformer/test_data/loss_kinetics_test_small_1-2_3_30.json",
        sampling_rate_loss=2,
        selection_method="adaptive"
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    for i, (frames, file_name, label) in enumerate(test_loader):
        breakpoint()

        output = model(frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'swin'],
                        help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='/home/reutemann/Dino-Video-Summarization-Transformer/checkpoints/TimeSformer_divST_8x32_224_K400.pyth'
, type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--lc_pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dataset', default="kinetics400", help='Dataset: ucf101 / hmdb51')
    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    # Model parameters
    parser.add_argument('--out_dim', default=768, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    
    args = parser.parse_args()

    timesformer_classification(args)
