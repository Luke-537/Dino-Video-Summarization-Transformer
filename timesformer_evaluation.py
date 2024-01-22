import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import av
import numpy as np
import torch
import logging

from transformers import AutoImageProcessor, TimesformerForVideoClassification
from utils.parser import parse_args, load_config
from datasets_custom import FrameSelectionLoader


def read_video_pyav(container, indices):
    """
    container (InputContainer): PyAV container.
    indices (List[int]): list of frame indices to decode.

    """

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

def evaluation():
    """
    Evaluating the frame selection model on Kinetics-400 using a pre-trained TimeSformer classification model
    """

    # setting config parameters
    args = parse_args()
    args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    config = load_config(args)
    config.DATA.PATH_TO_DATA_DIR = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations"
    config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized"
    config.DATASET = "Kinetics"
    config.LOSS_FILE = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values/loss_kinetics_test_4_3_30.json"

    # initialising the test dataset, returning the selected indices for the videos
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

    # initialising the pretrained model and the image processor
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained("timesformer_finetuning")
    model.cuda()

    # creating logging file for the results
    logging.basicConfig(filename='eval_logs/k400_adaptive_finetuned.log', level=logging.INFO)

    # counting the number of total and correct predictions
    total_pred = 0
    correct_pred = 0

    # iterating over each video in the dataset
    for i in range(len(dataset)):
        # printing in console every 10 steps
        if i % 10 == 0 and i != 0:
            print(str(i) + "/" + str(len(dataset)) + "   ")
            print("Accuracy: " + str(correct_pred / total_pred * 100) + "%")

        # load the video using the list of selected indices
        container = av.open("/graphics/scratch2/students/reutemann/kinetics-dataset/k400/test/" + dataset[i][2])
        video = read_video_pyav(container, dataset[i][0])

        # add padding for size mismatch
        if video.shape[0] != 16:
            padding_value = 16 - video.shape[0]
            padding = ((padding_value, 0), (0, 0), (0, 0), (0, 0))
            video = np.pad(video, pad_width=padding, mode='constant', constant_values=0)

        # prepare video for the model
        inputs = processor(list(video), return_tensors="pt")
        inputs.to('cuda:0')
        
        # generating outputs
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # model predicts one of the 400 Kinetics-400 classes
        predicted_class_idx = logits.argmax(-1).item()

        if predicted_class_idx == dataset[i][1]:
            correct_pred = correct_pred + 1
        total_pred = total_pred + 1

        # logging every 250 steps
        if i % 250 == 0: 
            logging.info(f"Sample {i}   Accuracy: {correct_pred / total_pred * 100}%   Correct Predictions: {correct_pred}   Total Predictions: {total_pred}")

    # logging at the end
    logging.info(f"Sample {i}   Accuracy: {correct_pred / total_pred * 100}%   Correct Predictions: {correct_pred}   Total Predictions: {total_pred}")

if __name__ == '__main__':
    evaluation()