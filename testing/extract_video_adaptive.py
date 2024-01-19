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
from testing.visualization import save_tensor_as_video, plot_loss
from torchvision.utils import save_image


def extract_video(cfg, video_path, loss_path, pre_sampling_rate, selection_method, out_path, save_frames=False):

    video, audio, info = io.read_video(video_path, pts_unit='sec')
    frames_unsampled = video.to(torch.float)

    with open(loss_path, 'r') as file:
        loss_dict = json.load(file)

    frames_sampled = frames_unsampled[::pre_sampling_rate]
    #frames = tensor_normalize(frames_sampled, cfg.DATA.MEAN, cfg.DATA.STD)

    # T H W C -> T C H W.
    frames = frames_sampled.permute(0, 3, 1, 2)
    #frames, _ = uniform_crop(frames, size=224, spatial_idx=1) # adjust params

    N = 8  # Number of frames to select

    file_name = os.path.basename(video_path)

    if selection_method == "adaptive":   
        # Get the file name and then the loss values
        key = os.path.splitext(file_name)[0]
        loss_list = loss_dict[key]

        #sharpening the values
        #loss_list = np.asarray(loss_list) ** 2
        loss_list = np.asarray(loss_list)

        if len(loss_list) > frames.size(0):
            loss_list = loss_list[:frames.size(0)]

        # min-max normalization
        pdf = (loss_list - loss_list.min()) / (loss_list.max() - loss_list.min())

        # Normalizing the loss values to create a PDF, might need to scale
        pdf = loss_list / np.sum(loss_list)

        # Create the CDF from the PDF
        cdf = np.cumsum(pdf)

        selected_frames = []
        indices = []
        for i in range(N):
            # Find the frame index corresponding to the quantile
            j = i / N
            cdf_array = np.asarray(cdf)
            idx = (np.abs(cdf_array - j)).argmin()
            selected_frames.append(frames[idx])
            indices.append(idx*pre_sampling_rate)

        frames = torch.stack(selected_frames)
        
    else:
        # only sample every n-th frame
        selected_frames = []
        interval = int(frames.size(0) / N)

        for i in range(N):
            selected_frames.append(frames[i*interval])

        frames = torch.stack(selected_frames)
        indices = None

    # T C H W -> C T H W.
    frames = frames.permute(1, 0, 2, 3)
    
    save_tensor_as_video(frames, out_path)

    if save_frames:
        key = os.path.splitext(file_name)[0]
        frames = frames.permute(1, 0, 2, 3) / 255
        for i in range(len(frames)):
            path = 'videos_sampled/'+ str(key) + '/' + str(i) + '.png'
            save_image(frames[:][i], path)

    return indices

if __name__ == '__main__':
    key = "giLxPCgLLqg_9_19"
    dataset = "MSVD"
    #dataset = "Kinetics"

    dir_path = "/home/reutemann/Dino-Video-Summarization-Transformer/videos_sampled/" + key
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if dataset == "MSVD":
        video_path = "/graphics/scratch/datasets/MSVD/YouTubeClips/" + key + ".avi"
        loss_path = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_msvd_4_3_30.json"
    else:
        loss_path = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_test_4_3_30.json"
        video_path = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test/" + key + ".mp4"

    args = parse_args()
    args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    cfg = load_config(args)
    out_path = dir_path + "/" + key + "_u.mp4"
    _ = extract_video(cfg, video_path, loss_path, 4, "uniform", out_path, save_frames=False)
    out_path = dir_path + "/" + key + "_a.mp4"
    indices = extract_video(cfg, video_path, loss_path, 4, "adaptive", out_path, save_frames=True)

    plot_loss(loss_path, 4, dir_path + "/" + key + ".png", key=key, selected_frames=indices)