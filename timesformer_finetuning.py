from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoImageProcessor, TimesformerForVideoClassification
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import Trainer, TrainingArguments
from datasets_custom import FrameSelectionLoader
from utils.parser import parse_args, load_config
import matplotlib.pyplot as plt

# Load pre-trained model
processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400", ignore_mismatched_sizes=True) # finetune 1-2 epoch
model.cuda()

# Ensuring model uses CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

args = parse_args()
args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
config = load_config(args)
config.DATA.PATH_TO_DATA_DIR = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations"
config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test"

dataset = FrameSelectionLoader(
    cfg=config,
    loss_file="/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_test_4_3_30.json",
    pre_sampling_rate=4,
    selection_method="adaptive",
    num_frames=16,
    augmentations=True,
    return_dict = True
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/home/reutemann/Dino-Video-Summarization-Transformer/timesformer_finetuning/results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="/home/reutemann/Dino-Video-Summarization-Transformer/timesformer_finetuning/logs",
    logging_steps=1000,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Your custom dataset
    eval_dataset=dataset
)

#breakpoint()

# Train the model
trainer.train()

# Save the model
model.save_pretrained("/home/reutemann/Dino-Video-Summarization-Transformer/timesformer_finetuning")

"""
training_loss = [log['loss'] for log in logs]  # Extract loss values from logs
plt.plot(training_loss, label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.show()
"""
