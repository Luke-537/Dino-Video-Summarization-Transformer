import os
import torch
import torch.utils.data
import torchvision.io as io

from datasets_custom.transform import uniform_crop
from datasets_custom.data_utils import tensor_normalize


class DinoLossLoader(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, local_clip_size, global_clip_size, sampling_rate):
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

        for i in range(self.global_clip_size*2):
            self.dummy_list.append(torch.zeros(3, 60, self.crop_size, self.crop_size))

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
            #self.cfg.DATA.PATH_TO_DATA_DIR, "{}_small.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []

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
        video, audio, info = io.read_video(self._path_to_videos[index], pts_unit='sec')
        frames_unsampled = video.to(torch.float)

        # only sample every n-th frame
        frames_sampled = frames_unsampled[::self.sampling_rate].to(torch.uint8)

        frames = tensor_normalize(frames_sampled, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        frames, _ = uniform_crop(frames, size=self.crop_size, spatial_idx=1) # adjust params

        views_list = get_views_of_video_same_size(
            frames,
            self.local_clip_size,
            self.global_clip_size,
        ) 

        # return an empty tensor if there is a size mismatch, later resulting in constant loss values
        if size_match(views_list):

            return views_list, self._path_to_videos[index], frames_sampled.permute(3, 0, 1, 2)
        else:

            return torch.stack(self.dummy_list), self._path_to_videos[index], frames_sampled.permute(3, 0, 1, 2)


    def __len__(self):

        return len(self._path_to_videos)
    

def size_match(list):
    same_size = True

    for tensor in list[1:]:
        if tensor.shape != list[0].shape or tensor.size(-2) != 224 or tensor.size(-1) != 224:
            same_size = False

    return same_size


def get_views_of_video_same_size(frames, local_size, global_size):
    #same as other function but views are the same size for batching
    loc = int(local_size / 2)
    glob = int(global_size / 2)

    views_list = []

    if len(frames) < global_size:
        global_size = len(frames)

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

        tensor_padded = torch.zeros(3, global_size, tensor_local.size(2), tensor_local.size(3))       
        tensor_padded[:, :local_size, :] = tensor_local

        views_list.append(tensor_padded)
        views_list.append(tensor_global)

    return torch.stack(views_list)
