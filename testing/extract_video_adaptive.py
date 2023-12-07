import sys
sys.path.append('/home/reutemann/Dino-Video-Summarization-Transformer')
import glob
import os
import random
import warnings
from PIL import Image
import torch
import torch.utils.data
import torchvision
import torchvision.io as io
from torchvision.transforms import functional as tf
import torch.nn.functional as F
import kornia
import json
import numpy as np
from utils.parser import parse_args, load_config

from datasets.transform import resize, uniform_crop, color_normalization
from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets.decoder import decode
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationDINO
from einops import rearrange
from testing.visualization import save_tensor_as_video


def extract_video(cfg, video_path, loss_path, pre_sampling_rate, selection_method, out_path):

    video, audio, info = io.read_video(video_path, pts_unit='sec')
    frames_unsampled = video.to(torch.float)

    with open(loss_path, 'r') as file:
        loss_dict = json.load(file)

    frames_sampled = frames_unsampled[::pre_sampling_rate]
    #frames = tensor_normalize(frames_sampled, cfg.DATA.MEAN, cfg.DATA.STD)

    # T H W C -> T C H W.
    frames = frames_sampled.permute(0, 3, 1, 2)
    #frames, _ = uniform_crop(frames, size=224, spatial_idx=1) # adjust params

    if selection_method == "adaptive":   
        # Get the file name and then the loss values
        file_name = os.path.basename(video_path)
        key = os.path.splitext(file_name)[0]
        loss_list = loss_dict[key]

        #breakpoint()

        #sharpening the values
        loss_list = np.asarray(loss_list) ** 2

        # Normalizing the loss values to create a PDF
        pdf = loss_list / np.sum(loss_list)

        # Create the CDF from the PDF
        cdf = np.cumsum(pdf)

        N = 32  # Number of frames to select
        selected_frames = []
        indices = []
        for i in range(N):
            # Find the frame index corresponding to the quantile
            j = i / N
            cdf_array = np.asarray(cdf)
            idx = (np.abs(cdf_array - j)).argmin()
            selected_frames.append(frames[idx])
            indices.append(idx)

        frames = torch.stack(selected_frames)
        
    else:
        # only sample every n-th frame
        N = int(frames.size(0) / 32)
        frames = frames[::N]

    # T C H W -> C T H W.
    frames = frames.permute(1, 0, 2, 3)
    
    save_tensor_as_video(frames, out_path)


if __name__ == '__main__':
    video_path = "/graphics/scratch/datasets/MSVD/YouTubeClips/B-Lsf7ZKf5c_10_25.avi"
    loss_path = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values/loss_msvd_2_3_30.json"
    args = parse_args()
    args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    cfg = load_config(args)
    out_path = "/home/reutemann/Dino-Video-Summarization-Transformer/videos_sampled/video_3_a_3.mp4"
    extract_video(cfg, video_path, loss_path, 2, "adaptive", out_path)
