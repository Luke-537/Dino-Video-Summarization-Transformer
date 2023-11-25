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

from datasets.transform import resize, uniform_crop, color_normalization
from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets.decoder import decode
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationDINO
from einops import rearrange

class FrameSelectionLoader(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, loss_file, sampling_rate_loss, selection_method="uniform"):
        self.cfg = cfg
        self.mode = mode

        self.sampling_rate_loss = sampling_rate_loss
        self.selection_method = selection_method

        self._video_meta = {}
        self._num_retries = 10

        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        self.crop_size = 224

        with open(loss_file, 'r') as file:
            self.loss_dict = json.load(file)


        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
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
        video, audio, info = io.read_video(self._path_to_videos[index], pts_unit='sec')
        frames_unsampled = video.to(torch.float)

        frames_sampled = frames_unsampled[::self.sampling_rate_loss]
        frames = tensor_normalize(frames_sampled, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        frames, _ = uniform_crop(frames, size=self.crop_size, spatial_idx=1) # adjust params

        if self.selection_method == "adaptive":
            breakpoint()
            
            # Get the file name and then the loss values
            file_name = os.path.basename(self._path_to_videos[index])
            key = os.path.splitext(file_name)[0]
            loss_list = self.loss_dict[key]

            # Normalizing the loss values to create a PDF
            pdf = loss_list / np.sum(loss_list)

            # Create the CDF from the PDF
            cdf = np.cumsum(pdf)

            N = 32  # Number of frames to select
            selected_frames = []
            indices = []
            for i in range(N):
                # Find the frame index corresponding to the quantile
                value = i / N
                array = np.asarray(cdf)
                idx = (np.abs(array - value)).argmin()
                selected_frames.append(frames[idx])
                indices.append(idx)

            frames = torch.stack(selected_frames)
            
        else:
            # only sample every n-th frame
            frames = frames[::10]

        breakpoint()

        # T C H W -> C T H W.
        frames = frames.permute(1, 0, 2, 3)

        return frames, self._path_to_videos[index]

    def __len__(self):

        return len(self._path_to_videos)
    
