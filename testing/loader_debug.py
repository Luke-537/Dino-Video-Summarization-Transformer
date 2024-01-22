import sys
sys.path.insert(0, '/home/reutemann/Dino-Video-Summarization-Transformer')

from utils.parser import parse_args, load_config
from datasets_custom import FrameSelectionLoader


def debug_dataloader():
    """
    Debugging script for the data loader.
    """
        
    args = parse_args()
    args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    config = load_config(args)

    if False:
        config.DATASET = "Kinetics"
        config.LOSS_FILE = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values/loss_kinetics_train_4_3_30.json"
        config.DATA.PATH_TO_DATA_DIR = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations"
        config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized"
    else:
        config.DATASET = "MSVD"
        config.LOSS_FILE = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values/loss_msvd_4_3_30.json"
        config.DATA.PATH_TO_DATA_DIR = "/home/reutemann/Dino-Video-Summarization-Transformer/MSVD"
        config.DATA.PATH_PREFIX = "/graphics/scratch/datasets/MSVD/YouTubeClips"

    dataset = FrameSelectionLoader(
        cfg=config,
        pre_sampling_rate=4,
        selection_method="adaptive",
        num_frames=8,
        augmentations=False
    )

    for i in range(len(dataset)):
        breakpoint()
        item = dataset[i]


if __name__ == '__main__':
    debug_dataloader()
