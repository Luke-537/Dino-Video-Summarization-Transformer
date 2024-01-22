import os
import torch
import torch.utils.data
import torchvision.io as io

from datasets_custom.transform import uniform_crop
from datasets_custom.data_utils import tensor_normalize


class DinoLossLoader(torch.utils.data.Dataset):
    """
    Video loader for Kinetics and MSVD that returns the local and global clips for each frame of a video using 
    a defined size for both, applies augmentations and also downsamples the video first. 
    """

    def __init__(self, cfg, mode, local_clip_size, global_clip_size, sampling_rate):
        """
        cfg (CfgNode): configs.
        mode (String): dataset split, one of "test", "train" or "val".
        local_clip_size (int): size of the local views.
        global_clip_size (int): size of the global views.
        sampling_rate (int): pre-sampling rate to downsample the video.
        """

        self.cfg = cfg
        self.mode = mode
        self.local_clip_size = local_clip_size
        self.global_clip_size = global_clip_size
        self.sampling_rate = sampling_rate
        self._video_meta = {}
        self._num_retries = 10
        self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS
        self.crop_size = 224
        self.dummy_list = []

        # defining a dummy return tensor for corrupted videos
        for i in range(self.global_clip_size*2):
            self.dummy_list.append(torch.zeros(3, 60, self.crop_size, self.crop_size))

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

        # only sample every n-th frame
        frames_sampled = frames_unsampled[::self.sampling_rate].to(torch.uint8)
        frames = tensor_normalize(frames_sampled, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)

        # center crop with size 224x224
        frames, _ = uniform_crop(frames, size=self.crop_size, spatial_idx=1) # adjust params

        # calculating the local and global clips for each frame
        views_list = get_views_of_video_same_size(
            frames,
            self.local_clip_size,
            self.global_clip_size,
        ) 

        # return the list of views, the file name and the whole video for debugging or
        # an empty tensor if there is a size mismatch, later resulting in constant loss values
        if size_match(views_list):
            return views_list, self._path_to_videos[index], frames_sampled.permute(3, 0, 1, 2)
        
        else:
            return torch.stack(self.dummy_list), self._path_to_videos[index], frames_sampled.permute(3, 0, 1, 2)


    def __len__(self):

        return len(self._path_to_videos)
    

def size_match(list):
    # check if the tensor sizes fit the original video
    same_size = True

    for tensor in list[1:]:
        if tensor.shape != list[0].shape or tensor.size(-2) != 224 or tensor.size(-1) != 224:
            same_size = False

    return same_size


def get_views_of_video_same_size(frames, local_size, global_size):
    # define as half the clip sizes to get the frames before and after the frame that is being computed
    loc = int(local_size / 2)
    glob = int(global_size / 2)
    views_list = []

    if len(frames) < global_size:
        global_size = len(frames)

    # iterating over all frames to retrieve the neighborhoods
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

        # T C H W -> C T H W.
        tensor_local = frames[j:k].permute(1, 0, 2, 3)
        tensor_global = frames[l:m].permute(1, 0, 2, 3)

        # padding the tensors for edge cases and for the local views
        tensor_padded = torch.zeros(3, global_size, tensor_local.size(2), tensor_local.size(3))       
        tensor_padded[:, :local_size, :] = tensor_local

        # append the tensors to the list of views
        views_list.append(tensor_padded)
        views_list.append(tensor_global)

    # converting the list to a tensor
    return torch.stack(views_list)
