import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import av
import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoImageProcessor, TimesformerForVideoClassification
from utils.parser import parse_args, load_config
from datasets_custom import FrameSelectionLoader
import logging


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


args = parse_args()
args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
config = load_config(args)
config.DATA.PATH_TO_DATA_DIR = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations"
config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized"
config.DATASET = "Kinetics"
config.LOSS_FILE = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_test_4_3_30.json"

dataset = FrameSelectionLoader(
    cfg=config,
    pre_sampling_rate=4, 
    selection_method="adaptive",
    num_frames=16,
    augmentations=False,
    return_type="Indices",
    mode="test"
)
print(f"Loaded dataset of length: {len(dataset)}")
#dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16)


processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("timesformer_finetuning_test_2")
model.cuda()

logging.basicConfig(filename='eval_logs/k400_adaptive_finetuned_test_2.log', level=logging.INFO)

total_pred = 0
correct_pred = 0

for i in range(len(dataset)):
    if i % 10 == 0 and i != 0:
        print(str(i) + "/" + str(len(dataset)) + "   ")
        print("Accuracy: " + str(correct_pred / total_pred * 100) + "%")

    container = av.open("/graphics/scratch2/students/reutemann/kinetics-dataset/k400/test/" + dataset[i][2]) #maybe try resized
    video = read_video_pyav(container, dataset[i][0])

    if video.shape[0] != 16:
        padding_value = 16 - video.shape[0]
        padding = ((padding_value, 0), (0, 0), (0, 0), (0, 0))
        video = np.pad(video, pad_width=padding, mode='constant', constant_values=0)

    # prepare video for the model
    inputs = processor(list(video), return_tensors="pt")
    inputs.to('cuda:0')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # model predicts one of the 400 Kinetics-400 classes
    predicted_class_idx = logits.argmax(-1).item()
    #print(model.config.id2label[predicted_label])
    if predicted_class_idx == dataset[i][1]:
        correct_pred = correct_pred + 1
    total_pred = total_pred + 1

    if i % 250 == 0: 
        logging.info(f"Sample {i}   Accuracy: {correct_pred / total_pred * 100}%   Correct Predictions: {correct_pred}   Total Predictions: {total_pred}")

logging.info(f"Sample {i}   Accuracy: {correct_pred / total_pred * 100}%   Correct Predictions: {correct_pred}   Total Predictions: {total_pred}")