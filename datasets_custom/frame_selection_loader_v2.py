import os
import torch
import torch.utils.data
import torchvision
import torchvision.io as io
from torchvision.transforms import functional as tf
import torch.nn.functional as F
import kornia
import json
import numpy as np

class FrameSelectionLoaderv2(torch.utils.data.Dataset):

    def __init__(self, cfg, loss_file, pre_sampling_rate, selection_method="uniform", num_frames=16):

        self.cfg = cfg

        self.pre_sampling_rate = pre_sampling_rate
        self.selection_method = selection_method

        self._video_meta = {}
        self._num_retries = 10

        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        self.num_frames = num_frames

        self.crop_size = 224

        with open(loss_file, 'r') as file:
            self.loss_dict = json.load(file)

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format("test")
            #self.cfg.DATA.PATH_TO_DATA_DIR, "{}_small.csv".format("test")
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
                        os.path.join("/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test", path)
                        #os.path.join("/graphics/scratch/datasets/MSVD/YouTubeClips", path)
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
        frames = frames_unsampled[::self.pre_sampling_rate].to(torch.uint8)

        # Number of frames to select
        N = self.num_frames 

        file_name = os.path.basename(self._path_to_videos[index])

        if self.selection_method == "adaptive":   
            # Get the raw file name and then the loss values
            key = os.path.splitext(file_name)[0]
            loss_list = self.loss_dict[key]

            #sharpening the values
            #loss_list = np.asarray(loss_list) ** 2
            loss_list = np.asarray(loss_list)

            # min-max normalization
            pdf = (loss_list - loss_list.min()) / (loss_list.max() - loss_list.min())

            # Normalizing the loss values to create a PDF, might need to scale
            pdf = loss_list / np.sum(loss_list)

            # Create the CDF from the PDF
            cdf = np.cumsum(pdf)

            indices = []
            for i in range(N):
                # Find the frame index corresponding to the quantile
                j = i / N
                cdf_array = np.asarray(cdf)
                idx = (np.abs(cdf_array - j)).argmin()
                idx_scaled = idx*self.pre_sampling_rate
                #indices.append(idx_scaled)

                if idx_scaled not in indices:
                    indices.append(idx_scaled)
                else:
                    temp = idx_scaled + self.pre_sampling_rate
                    search = True
                    while search:
                        if temp not in indices and temp < frames_unsampled.size(0):
                            indices.append(temp)
                            search = False
                        elif temp >= frames_unsampled.size(0):
                            indices.append(temp - self.pre_sampling_rate)
                            search = False
                        else:
                            temp = temp + self.pre_sampling_rate
            
        else:
            # only sample every n-th frame
            indices = []
            interval = int(frames_unsampled.size(0) / N)

            for i in range(N):
                indices.append(i*interval)

        if len(indices) != N:
            for i in range(N - len(indices)):
                indices.append(frames.size(0)-1)

        return indices, self._labels[index], file_name
    

    def __len__(self):

        return len(self._path_to_videos)