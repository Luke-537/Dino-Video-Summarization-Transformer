import sys
sys.path.insert(0, '/home/reutemann/Dino-Video-Summarization-Transformer')
import torchvision.io as io
from visualization import save_tensor_as_video
from utils.parser import parse_args, load_config
from tqdm import tqdm
from datasets_custom import Kinetics, DinoLossLoader, KineticsFinetune, FrameSelectionLoader, FrameSelectionLoaderv2
import torch

if __name__ == '__main__':

    args = parse_args()
    args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    config = load_config(args)
    #config.DATA.PATH_TO_DATA_DIR = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations"
    config.DATA.PATH_TO_DATA_DIR = "/home/reutemann/Dino-Video-Summarization-Transformer/MSVD"
    #config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/train"
    #config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test"
    #config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/val"
    config.DATA.PATH_PREFIX = "/graphics/scratch/datasets/MSVD/YouTubeClips"
    #dataset = Kinetics(cfg=config, mode="train", num_retries=10)
    #dataset = Kinetics(cfg=config, mode="val", num_retries=10)
    #dataset = KineticsFinetune(cfg=config, mode="train", num_retries=10, get_flow=False)
    #dataset = DinoLossLoader(cfg=config, mode="test", local_clip_size=3, global_clip_size=60, sampling_rate=8)
    dataset = FrameSelectionLoader(cfg=config, loss_file="/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_msvd_4_3_30.json", pre_sampling_rate=4, selection_method="adaptive", num_frames=8, augmentations=False)
    #dataset = FrameSelectionLoaderv2(cfg=config, loss_file="/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_test_4_3_30.json", pre_sampling_rate=4, selection_method="adaptive", num_frames=16)
    print(f"Loaded dataset of length: {len(dataset)}")
    #dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16)

    for i in range(len(dataset)):
        breakpoint()
        sds = dataset[i]

    """
    for (images, label, index, meta) in dataloader:

        breakpoint()
        save_tensor_as_video(images[0][6], "/home/reutemann/Dino-Video-Summarization-Transformer/video.mp4")
    """
