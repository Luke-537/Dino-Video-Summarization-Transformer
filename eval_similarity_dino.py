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

from datasets import UCF101, HMDB51, Kinetics, KineticsCustom
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config


def eval_dino(args, video_path):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    
    config = load_config(args)
    
    # Load the basic model class
    model = get_vit_base_patch16_224(cfg=config, no_head=False)

    # Load the pretrained checkpoint
    ckpt = torch.load(args.pretrained_weights)
    renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}

    # Load the corresponding state dict and switch to eval mode
    model.load_state_dict(renamed_checkpoint, strict=False)
    model.cuda()
    model.eval()

    """
    # Printing the state dict

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    """

    # Load teacher and student model class
    student = get_vit_base_patch16_224(cfg=config, no_head=False)
    teacher = get_vit_base_patch16_224(cfg=config, no_head=False)
    student.load_state_dict(renamed_checkpoint, strict=False)
    teacher.load_state_dict(renamed_checkpoint, strict=False)

    student, teacher = student.cuda(), teacher.cuda()

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=False)

    local_clip_size = 3
    global_clip_size = 60
    
    
    # load test dataset
    dataset_test = KineticsCustom(
        cfg=config,
        mode="test",
        local_clip_size=local_clip_size, 
        global_clip_size=global_clip_size
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Instantiate dino loss
    dino_loss = DINOLoss(
        args.out_dim,
        2,  # total number of crops = 1 global crops +  1 local crop
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        global_crops=1,
        two_token=config.MODEL.TWO_TOKEN
    )
    dino_loss = dino_loss.cuda()

    for i, (local_views, global_views, file_name) in enumerate(test_loader):
        #breakpoint()

        loss = []
        for x in range(len(local_views)):
            
            #save_tensor_as_video(global_views[x][0])

            print((x+1), "/", len(local_views))

            local_view = local_views[x].cuda(non_blocking=True)
            global_view = global_views[x].cuda(non_blocking=True)

            #local_view = local_views[n*4:(n+1)*4].cuda(non_blocking=True)
            #global_view = global_views[n*4:(n+1)*4].cuda(non_blocking=True)

            with torch.no_grad():
                student_output = student(local_view)
                teacher_output = teacher(global_view)

            for y in range(len(student_output)):
                loss.append(dino_loss(student_output[y], teacher_output[y]).item())

        #export_loss(loss, file_name[0], 'loss_k400_resized_test/loss_output_fintuned_3_60_test.json')

        break


def save_tensor_as_video(tensor):
    tensor = tensor.permute(1, 2, 3, 0)

    io.write_video('test_videos/video_test_2.mp4', tensor, fps=30)


def export_loss(loss_list, video_path, file_path):
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


def create_correllation_matrix(loss_list):
    correlation_matrix = np.zeros((len(loss_list), len(loss_list)))

    for i in range(len(loss_list)):
        for j in range(len(loss_list)):

            correlation_matrix[i][j] = i + j

    return correlation_matrix


# DINO Loss that only takes a 1-dimensional tensor as an input for student_output and teacher_output 
# before it needed at least 2 local views
# also removed the check if model is two token for readability
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, global_crops=2, two_token=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = ncrops
        self.global_crops = global_crops
        self.two_token = two_token
        self.register_buffer("center", torch.zeros(1, out_dim))
        # Just using the temp, not a schedule beacause no training
        self.teacher_temp_schedule = teacher_temp

    # removed chunking and cumulative loss calculation
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = 0
        n_loss_terms = 0
        
        # Calculate the teacher and student losses for the entire tensors
        q_teacher = F.softmax((teacher_output - self.center) / self.teacher_temp_schedule, dim=-1)
        q_student = student_output / self.student_temp
        total_loss += torch.sum(-q_teacher * F.log_softmax(q_student, dim=-1), dim=-1).mean()
        n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    # Do I even need to calculate the centering? Does it make a difference when not backpropagating?
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        if isinstance(teacher_output, (tuple, list)):
            # Update centers for each view (if applicable)
            for i, teacher_view in enumerate(teacher_output):
                batch_center = torch.sum(teacher_view, dim=0, keepdim=True)
                dist.all_reduce(batch_center)
                batch_center = batch_center / (len(teacher_view) * dist.get_world_size())
                self.center[i, :] = self.center[i, :] * self.center_momentum + batch_center * (1 - self.center_momentum)
        else:
            # Update center for the entire tensor
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


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

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    args = parser.parse_args()
    video_path = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test/__7xZtQ9fz0_000140_000150.mp4"

    eval_dino(args, video_path)
