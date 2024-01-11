from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoImageProcessor, TimesformerForVideoClassification
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from transformers import Trainer, TrainingArguments
from datasets_custom import FrameSelectionLoader
from utils.parser import parse_args, load_config
import matplotlib.pyplot as plt
import json

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
#config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test"

dataset_train = FrameSelectionLoader(
    cfg=config,
    loss_file="/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_train_4_3_30.json",
    pre_sampling_rate=4,
    selection_method="adaptive",
    num_frames=16,
    augmentations=True,
    return_dict = True,
    mode = "train"
)

dataset_val = FrameSelectionLoader(
    cfg=config,
    loss_file="/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_val_4_3_30.json",
    pre_sampling_rate=4,
    selection_method="adaptive",
    num_frames=16,
    augmentations=True,
    return_dict = True,
    mode = "val"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/graphics/scratch2/students/reutemann/timesformer_finetuning_test/results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="/graphics/scratch2/students/reutemann/timesformer_finetuning_test/logs",
    logging_steps=100,
    evaluation_strategy="epoch",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,  # Your custom dataset
    eval_dataset=dataset_val
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("/home/reutemann/Dino-Video-Summarization-Transformer/timesformer_finetuning_new")

#breakpoint()

# Assume 'trainer' is your Hugging Face Trainer object after training
log_history = trainer.state.log_history

# Extract loss values
training_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
validation_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]

# Plotting
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

with open('eval_logs/training_log_history.json', 'w') as file:
    json.dump(log_history, file)

# Save the plot
plt.savefig('eval_logs/finetuning_loss.png')  # Saves the plot as a PNG file

