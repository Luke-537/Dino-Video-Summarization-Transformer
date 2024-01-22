import os
import torch
import torch.utils.data
import torchvision.io as io
import json
import numpy as np

from datasets_custom.transform import uniform_crop
from datasets_custom.data_utils import tensor_normalize


class FrameSelectionLoader(torch.utils.data.Dataset):
    """
    Video loader for Kinetics and MSVD that uniformly samples N frames from each video or
    adaptively sampled N frames using pre-calculated loss values. It then returns either the
    video tensors or a list of the selected indices.
    """

    def __init__(self, cfg, pre_sampling_rate, selection_method="uniform", num_frames=8, augmentations=False, return_type="Tensor", mode="test"):
        """
        cfg (CfgNode): configs.
        pre_sampling_rate (int): pre-sampling rate with which the loss values have been calculated.
        selection_method (String): "uniform" or "adaptive".
        num_frames (int): number of frames to select.
        augmentations (Bool): whether to apply augmentations.
        return_type (String): in which form the values should be returned.
        mode (String): dataset split, one of "test", "train" or "val".
        """

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

        # open JSON file containing loss values
        with open(cfg.LOSS_FILE, 'r') as file:
            self.loss_dict = json.load(file)

        # open CSV file containing video names
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
            #self.cfg.DATA.PATH_TO_DATA_DIR, "{}_small.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []

        # append paths to videos to a list
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
        N = self.num_frames  # number of frames to select
        file_name = os.path.basename(self._path_to_videos[index])

        # applying augmentations only if set to True in parameters
        if self.augmentations:
            frames_sampled = frames_unsampled[::self.pre_sampling_rate].to(torch.uint8) # pre-sampling the video
            frames = tensor_normalize(frames_sampled, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

            # T H W C -> T C H W.
            frames = frames.permute(0, 3, 1, 2)

            # center crop with size 224x224
            frames, _ = uniform_crop(frames, size=self.crop_size, spatial_idx=1)

        else:
            frames = frames_unsampled[::self.pre_sampling_rate].to(torch.uint8) # pre-sampling the video

            # T H W C -> T C H W.
            frames = frames.permute(0, 3, 1, 2)

        if self.selection_method == "adaptive":   
            # get the file name and then the loss values
            key = os.path.splitext(file_name)[0]
            loss_list = self.loss_dict[key]

            # optionally sharpening the values
            # loss_list = np.asarray(loss_list) ** 2
            loss_list = np.asarray(loss_list)

            # preventing the case that the loss list is longer than the number of frames in the video
            if len(loss_list) > frames.size(0):
                loss_list = loss_list[:frames.size(0)]

            # min-max normalization
            pdf = (loss_list - loss_list.min()) / (loss_list.max() - loss_list.min())

            # normalizing the loss values to create a PDF
            pdf = loss_list / np.sum(loss_list)

            # creating the CDF from the PDF
            cdf = np.cumsum(pdf)

            selected_frames = []
            indices = []

            # selecting N indices
            for i in range(N):
                # finding the frame index corresponding to the quantile
                j = i / N
                cdf_array = np.asarray(cdf)
                idx = (np.abs(cdf_array - j)).argmin()

                # scaling the index to the unsampled video length
                idx_scaled = idx*self.pre_sampling_rate

                if idx_scaled not in indices:
                    indices.append(idx_scaled)
                    selected_frames.append(frames[idx])

                # if the index has already been selected, try again for the next possible index until the end of the video
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

            # creating a tensor from the list of frames
            frames = torch.stack(selected_frames)
            
        else:
            # sampling N frames uniformly
            selected_frames = []
            indices = []
            interval = int(frames.size(0) / N)

            for i in range(N):
                selected_frames.append(frames[i*interval])
                indices.append(i*interval)

            frames = torch.stack(selected_frames)

        # catching the case where the inices listt is shorter than N
        if len(indices) != N:
            for i in range(N - len(indices)):
                indices.append(frames.size(0)-1)

        # T C H W -> C T H W.
        frames = frames.permute(1, 0, 2, 3)

        # returning the list of indices, the class label and file name
        if self.return_type == "Indices":
            return indices, self._labels[index], file_name
        
        # returning a dictionary containing the tensor and label
        elif self.return_type == "Dict":
            tensor_empty = torch.zeros([3, N, 224, 224])
            if frames.shape != tensor_empty.shape:
                frames = tensor_empty # return padded tensor

            data_dict = {
                'pixel_values': frames.permute(1, 0, 2, 3),
                'label': self._labels[index]
            }
            return data_dict
        
        # returning the tensor, label, file name and metadata
        else:
            return frames, self._labels[index], file_name, self._video_meta
        

    def __len__(self):

        return len(self._path_to_videos)
