import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import av
import numpy as np
import torch
from utils.parser import parse_args, load_config
from datasets import FrameSelectionLoaderv2
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

device = "cuda"

args = parse_args()
args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
config = load_config(args)
config.DATA.PATH_TO_DATA_DIR = "/home/reutemann/Dino-Video-Summarization-Transformer/MSVD"
config.DATA.PATH_PREFIX = "/graphics/scratch/datasets/MSVD/YouTubeClips"

dataset = FrameSelectionLoaderv2(
    cfg=config,
    mode="test",
    loss_file="/home/reutemann/Dino-Video-Summarization-Transformer/loss_values/loss_msvd_2_3_30.json",
    pre_sampling_rate=2, 
    selection_method="uniform",
    num_frames=32
)
print(f"Loaded dataset of length: {len(dataset)}")
#dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16)

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

# load video
video_path = "/graphics/scratch/datasets/MSVD/YouTubeClips/LYSPQqUvNO0_43_57.avi"
container = av.open(video_path)

# extract evenly spaced frames from video
seg_len = container.streams.video[0].frames
clip_len = model.config.encoder.num_frames
indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
frames = []
container.seek(0)
for i, frame in enumerate(container.decode(video=0)):
    if i in indices:
        frames.append(frame.to_ndarray(format="rgb24"))

# generate caption
gen_kwargs = {
    "min_length": 10, 
    "max_length": 20, 
    "num_beams": 8,
}
pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
tokens = model.generate(pixel_values, **gen_kwargs)
caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
print(caption) # A man and a woman are dancing on a stage in front of a mirror.