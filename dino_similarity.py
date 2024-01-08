# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import math
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

from datasets_custom import UCF101, HMDB51, Kinetics, DinoLossLoader
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
from testing.visualization import save_tensor_as_video


def dino_similarity(args, video_path):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    
    config = load_config(args)

    # Load the pretrained checkpoint
    ckpt_teacher = torch.load(args.pretrained_weights)#["teacher"]
    ckpt_student = torch.load(args.pretrained_weights)#["student"]
    renamed_checkpoint_teacher = {x[len("backbone."):]: y for x, y in ckpt_teacher.items() if x.startswith("backbone.")}
    renamed_checkpoint_student = {x[len("backbone."):]: y for x, y in ckpt_student.items() if x.startswith("backbone.")}

    # Load teacher and student model class, the corresponding state dict and switch to eval mode
    student = get_vit_base_patch16_224(cfg=config, no_head=False)
    teacher = get_vit_base_patch16_224(cfg=config, no_head=False)
    student.load_state_dict(renamed_checkpoint_student, strict=False)
    teacher.load_state_dict(renamed_checkpoint_teacher, strict=False)

    """
    # Printing the state dict

    print("Model's state_dict:")
    for param_tensor in teacher.state_dict():
        print(param_tensor, "\t", teacher.state_dict()[param_tensor].size())

    #print(teacher.keys())

    breakpoint()

    """

    student, teacher = student.eval(), teacher.eval()
    student, teacher = student.cuda(), teacher.cuda()

    # remove?
    #student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)
    #teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=False)

    local_clip_size = 3
    global_clip_size = 30
    sampling_rate = 4
    
    # load test dataset
    dataset_test = DinoLossLoader(
        cfg=config,
        mode="train",
        local_clip_size=local_clip_size, 
        global_clip_size=global_clip_size,
        sampling_rate=sampling_rate
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        #batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        #collate_fn=collate_fn_custom,
    )

    # Instantiate dino loss
    dino_loss = DINOLoss(
        out_dim=768,
        teacher_temp=0.02,
        student_temp=0.3,
    ).cuda()

    for i, (views_list, file_name, video) in enumerate(test_loader):

        #breakpoint()

        print(i+1, "/" ,len(test_loader))

        views = torch.squeeze(views_list, 0)

        loss = []
        batch = 0
        for x in range(math.ceil(len(views)/args.batch_size_per_gpu)):
            
            batch_new = batch + args.batch_size_per_gpu

            local_views = views[batch:batch_new][::2, :, :local_clip_size, :, :].cuda(non_blocking=True)
            global_views = views[batch:batch_new][1::2].cuda(non_blocking=True)

            #save_tensor_as_video(global_views[0].cpu(), "video_test_global.mp4")
            #save_tensor_as_video(local_views[0].cpu(), "video_test_local.mp4")

            with torch.no_grad():
                student_output = student(local_views)
                teacher_output = teacher(global_views)

            for y in range(len(student_output)):
                loss.append(dino_loss.forward(student_output[y], teacher_output[y]).item())

            batch = batch_new

        export_loss(loss, file_name[0])
        


def export_loss(loss_list, video_path):
    file_path = 'loss_values_new/loss_kinetics_train-3_4_3_30.json' 
    video_name = os.path.basename(video_path)
    video_name_without_extension, extension = os.path.splitext(video_name)

    video_dict = {video_name_without_extension: loss_list}

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
                    
        # Update data
        merged_dict = data.copy()
        merged_dict.update(video_dict)

        with open(file_path, 'w') as file:
            json.dump(merged_dict, file)

    else:
        with open(file_path, 'w') as file:
            json.dump(video_dict, file)

"""
def collate_fn_custom(batch):
    # Unzip the batch to separate the lists of tensors and the list of labels
    list_of_tensors, labels, video = zip(*batch)

    batched_list = []

    i = 0
    while i < len(list_of_tensors[0]):
        j = i + args.batch_size_per_gpu
        batched_list.append(torch.stack(list_of_tensors[0][i:j]))
        i = j

    return torch.stack(batched_list), labels, video
"""

# DINO Loss that only takes a 1-dimensional tensor as an input for student_output and teacher_output 
# before it needed at least 2 local views
# also removed the check if model is two token for readability
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp, 
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp

    # removed chunking and cumulative loss calculation
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # Calculate the teacher and student losses for the entire tensors
        p_teacher = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        p_student = student_output / self.student_temp
        total_loss = torch.sum(-p_teacher * F.log_softmax(p_student, dim=-1), dim=-1).mean()
        
        return total_loss


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
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
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
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dataset', default="ucf101", help='Dataset: ucf101 / hmdb51')
    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_60x16_224.yaml")
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
    video_path = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test/__7xZtQ9fz0_000140_000150.mp4"

    dino_similarity(args, video_path)
