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

from datasets.transform import resize, uniform_crop, color_normalization
from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets.decoder import decode
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationDINO
from einops import rearrange

class KineticsCustom(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, local_clip_size, global_clip_size):
        self.cfg = cfg
        self.mode = mode

        self.local_clip_size = local_clip_size
        self.global_clip_size = global_clip_size

        self._video_meta = {}
        self._num_retries = 10

        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        self.dummy_list = []
        for i in range(self.global_clip_size*2):
            self.dummy_list.append(torch.zeros(3, 60, 150, 150))

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
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        print(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    
    def __getitem__(self, index):
        frames = extract_frames_single_video(self._path_to_videos[index])

        # Perform normalization. NO COLOR
        frames = color_normalization(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        frames, _ = uniform_crop(frames, size=150, spatial_idx=1) # adjust params   
            
        views_list = get_views_of_video_same_size(
            frames,
            self.local_clip_size,
            self.global_clip_size,
        ) 

        if size_mismatch(views_list):

            return self.dummy_list, self._path_to_videos[index]
        else:

            return views_list, self._path_to_videos[index]


    def __len__(self):

        return len(self._path_to_videos)
    

def size_mismatch(list):
    first_tensor_size = list[0].shape
    same_size = True

    for tensor in list[1:]:
        if tensor.shape != first_tensor_size:
            same_size = False
    
    return same_size


def extract_frames_single_video(video_path):
    video, audio, info = io.read_video(video_path, pts_unit='sec')

    video = video.permute(0, 3, 1, 2)

    tensor_resized = torch.stack([tf.resize(frame, [224, 224]) for frame in video])

    return tensor_resized.to(torch.float)


def get_views_of_video_same_size(frames, local_size, global_size):
    #same as other function but views are the same size for batching
    loc = int(local_size / 2)
    glob = int(global_size / 2)

    views_list = []

    for i in range(len(frames)):
        j = i-loc
        k = i+loc+1
        l = i-glob
        m = i+glob

        if j < 0:
            j = 0
            k = local_size

        if k >= len(frames):
            k = len(frames)
            j = len(frames)-local_size

        if l < 0:
            l = 0
            m = global_size

        if m >= len(frames):
            m = len(frames)
            l = len(frames)-global_size

        tensor_local = frames[j:k].permute(1, 0, 2, 3)
        tensor_global = frames[l:m].permute(1, 0, 2, 3)

        tensor_padded = torch.zeros(3, global_size, tensor_local.size(2), tensor_local.size(3))       
        tensor_padded[:, :local_size, :] = tensor_local

        views_list.append(tensor_padded)
        views_list.append(tensor_global)

    return views_list


"""

def get_views_of_video(frames, local_size, global_size):
    frames = frames.permute(1, 0, 2, 3)

    loc = int(local_size / 2)
    glob = int(global_size / 2)

    local_views = []
    global_views = []

    for i in range(len(frames)):
        j = i-loc
        k = i+loc+1
        l = i-glob
        m = i+glob+1

        if j < 0:
            j = 0

        if k >= len(frames):
            k = len(frames)

        if l < 0:
            l = 0

        if m >= len(frames):
            m = len(frames)

        local_views.append(frames[j:k].permute(1, 0, 2, 3))
        global_views.append(frames[l:m].permute(1, 0, 2, 3))

    return local_views, global_views



# load single video as test
tensor_test = extract_frames_single_video(video_path)

# get local and global views for each frame of the video
local_views, global_views = get_views_of_video(tensor_test, local_clip_size, global_clip_size)

# is stretched video right?
#save_tensor_as_video(tensor_test)

    
def get_views_first_frame(frames):
    local_view = frames.permute(1, 0, 2, 3)
    local_view = local_view[:3].permute(1, 0, 2, 3)

    global_view = frames.permute(1, 0, 2, 3)
    global_view = global_view[:30].permute(1, 0, 2, 3)

    return local_view, global_view
    

"""