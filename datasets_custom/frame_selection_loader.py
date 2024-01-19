# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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

from datasets_custom.transform import resize, uniform_crop, color_normalization
from datasets_custom.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets_custom.decoder import decode
from datasets_custom.video_container import get_video_container
from datasets_custom.transform import VideoDataAugmentationDINO
from einops import rearrange
import torchvision.transforms as transforms

class FrameSelectionLoader(torch.utils.data.Dataset):

    def __init__(self, cfg, pre_sampling_rate, selection_method="uniform", num_frames=8, augmentations=False, return_type="Tensor", mode="test"):

        self.cfg = cfg
        self.mode = mode

        self.pre_sampling_rate = pre_sampling_rate
        self.selection_method = selection_method

        self._video_meta = {}
        self._num_retries = 10

        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        self.num_frames = num_frames

        self.crop_size = 224

        self.augmentations = augmentations

        self.return_type = return_type

        with open(cfg.LOSS_FILE, 'r') as file:
            self.loss_dict = json.load(file)

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
            #self.cfg.DATA.PATH_TO_DATA_DIR, "{}_small.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                        len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                        == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    if self.cfg.DATASET == "Kinetics":
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, mode,  path)
                        )
                    else:
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )

                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load data split {} from {}".format(
            self._split_idx, path_to_file
        )
        print(
            "Constructing dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    
    def __getitem__(self, index):
        video, _, _ = io.read_video(self._path_to_videos[index], pts_unit='sec')
        frames_unsampled = video.to(torch.float)
        N = self.num_frames  # Number of frames to select
        file_name = os.path.basename(self._path_to_videos[index])

        if self.augmentations:
            frames_sampled = frames_unsampled[::self.pre_sampling_rate].to(torch.uint8)
            frames = tensor_normalize(frames_sampled, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

            # T H W C -> T C H W.
            frames = frames.permute(0, 3, 1, 2)

            frames, _ = uniform_crop(frames, size=self.crop_size, spatial_idx=1) # adjust params
        else:
            frames = frames_unsampled[::self.pre_sampling_rate].to(torch.uint8)

            # T H W C -> T C H W.
            frames = frames.permute(0, 3, 1, 2)

        if self.selection_method == "adaptive":   
            # Get the file name and then the loss values
            key = os.path.splitext(file_name)[0]
            loss_list = self.loss_dict[key]

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
                idx_scaled = idx*self.pre_sampling_rate

                if idx_scaled not in indices:
                    indices.append(idx_scaled)
                    selected_frames.append(frames[idx])
                else:
                    temp = idx_scaled + self.pre_sampling_rate
                    search = True
                    while search:
                        if temp not in indices and temp < frames_unsampled.size(0):
                            indices.append(temp)
                            selected_frames.append(frames[idx])
                            search = False
                        elif temp >= frames_unsampled.size(0):
                            indices.append(temp - self.pre_sampling_rate)
                            selected_frames.append(frames[-1])
                            search = False
                        else:
                            temp = temp + self.pre_sampling_rate

            frames = torch.stack(selected_frames)
            
        else:
            # only sample every n-th frame
            selected_frames = []
            indices = []
            interval = int(frames.size(0) / N)

            for i in range(N):
                selected_frames.append(frames[i*interval])
                indices.append(i*interval)

            frames = torch.stack(selected_frames)

        if len(indices) != N:
            for i in range(N - len(indices)):
                indices.append(frames.size(0)-1)

        if self.mode == "train":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.25), # p is the probability of the image being flipped
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), # rotation and translation
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # color jittering
            ])
            
            # Apply the transform to your PyTorch tensor (assuming 'image' is your tensor)
            frames = transform(frames)

        # T C H W -> C T H W.
        frames = frames.permute(1, 0, 2, 3)

        if self.return_type == "Indices":
            return indices, self._labels[index], file_name
           
        elif self.return_type == "Dict":
            tensor_empty = torch.zeros([3, N, 224, 224])
            if frames.shape != tensor_empty.shape:
                #return padded tensor
                frames = tensor_empty
            data_dict = {
                'pixel_values': frames.permute(1, 0, 2, 3),
                'label': self._labels[index]
            }
            return data_dict
        else:
            return frames, self._labels[index], file_name, self._video_meta
            """
            if self.augmentations:
                tensor_empty = torch.zeros([3, N, 224, 224])
                if frames.shape != tensor_empty.shape: # switch to padding
                    frames = tensor_empty

                return frames, self._labels[index], file_name, self._video_meta
                
            else:
                return frames, self._labels[index], file_name, self._video_meta
            """

    def __len__(self):

        return len(self._path_to_videos)
