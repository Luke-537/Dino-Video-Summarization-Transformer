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

        # Perform color normalization.
        frames = color_normalization(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

        #augmentation = VideoDataAugmentationDINO()

        cropped_frames, _ = uniform_crop(frames, size=150, spatial_idx=1) # adjust params
   
        #frames = resize(frames, 150) # do i still need this after cropping?
        
            
        local_views, global_views = get_views_of_video_same_size(
            cropped_frames,
            self.local_clip_size,
            self.global_clip_size,
        )

        return local_views, global_views, self._path_to_videos[index]


    def __len__(self):

        return len(self._path_to_videos)


def extract_frames_single_video(video_path):
    video, audio, info = io.read_video(video_path, pts_unit='sec')

    video = video.permute(0, 3, 1, 2)

    tensor_resized = torch.stack([tf.resize(frame, [224, 224]) for frame in video])

    return tensor_resized.to(torch.float)


def get_views_of_video_same_size(frames, local_size, global_size):
    #same as other function but views are the same size for batching
    #frames = frames.permute(1, 0, 2, 3)

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

        target = torch.zeros(3, global_size, tensor_local.size(2), tensor_local.size(3))       
        target[:, :local_size, :] = tensor_local

        local_views.append(target)
        global_views.append(tensor_global)

    return local_views, global_views


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

"""