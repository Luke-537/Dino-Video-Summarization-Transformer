import argparse
import json
import math
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch import nn
from datasets_custom import DinoLossLoader
from models import get_vit_base_patch16_224
from utils import utils
from utils.parser import load_config


def dino_similarity(args, local_clip_size, global_clip_size, sampling_rate, file_path):
    cudnn.benchmark = True
    
    config = load_config(args)

    # Load the pretrained checkpoint
    ckpt = torch.load(args.pretrained_weights)
    renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}

    # Load teacher and student model class, the corresponding state dict and switch to eval mode
    model = get_vit_base_patch16_224(cfg=config, no_head=False)
    model.load_state_dict(renamed_checkpoint, strict=False)

    model= model.eval()
    model = model.cuda()
    
    # load test dataset
    dataset_test = DinoLossLoader(
        cfg=config,
        mode="test",
        local_clip_size=local_clip_size, 
        global_clip_size=global_clip_size,
        sampling_rate=sampling_rate
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Instantiate dino loss
    dino_loss = DINOLoss(
        out_dim=768,
        teacher_temp=0.02,
        student_temp=0.3,
    ).cuda()

    for i, (views_list, file_name, video) in enumerate(test_loader):
        print(i+1, "/" ,len(test_loader))
        views = torch.squeeze(views_list, 0)
        loss = []
        batch = 0

        for j in range(math.ceil(len(views)/args.batch_size_per_gpu)):
            
            batch_new = batch + args.batch_size_per_gpu

            local_views = views[batch:batch_new][::2, :, :local_clip_size, :, :].cuda(non_blocking=True)
            global_views = views[batch:batch_new][1::2].cuda(non_blocking=True)

            with torch.no_grad():
                student_output = model(local_views)
                teacher_output = model(global_views)

            for k in range(len(student_output)):
                loss.append(dino_loss.forward(student_output[k], teacher_output[k]).item())

            batch = batch_new

        export_loss(loss, file_name[0], file_path)
        


def export_loss(loss_list, video_path, file_path): 
    video_name = os.path.basename(video_path)
    video_name_without_extension, _ = os.path.splitext(video_name)

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
        #Cross-entropy between softmax outputs of the teacher and student networks.
        # Calculate the teacher and student losses for the entire tensors
        p_teacher = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        p_student = student_output / self.student_temp
        total_loss = torch.sum(-p_teacher * F.log_softmax(p_student, dim=-1), dim=-1).mean()
        
        return total_loss


if __name__ == '__main__':
    # Model parameters for the pre-trained SVT
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

    # parameters for the loss calculation, inlcuding the clips sizes, initial sampling rate and loss output path
    args = parser.parse_args()
    local_clip_size = 3
    global_clip_size = 30
    sampling_rate = 4
    file_path = 'loss_values/loss_kinetics_test_4_3_30.json'

    dino_similarity(args, local_clip_size, global_clip_size, sampling_rate, file_path)
