import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import matplotlib.pyplot as plt
import json

from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import Trainer, TrainingArguments
from datasets_custom import FrameSelectionLoader
from utils.parser import parse_args, load_config


def finetuning():
    # Load pre-trained model
    model = TimesformerForVideoClassification.from_pretrained("timesformer_finetuning", ignore_mismatched_sizes=True)
    model.cuda()

    # Ensuring model uses CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    args = parse_args()
    args.cfg_file = "/home/reutemann/Dino-Video-Summarization-Transformer/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    config = load_config(args)
    config.DATA.PATH_TO_DATA_DIR = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations"
    config.DATA.PATH_PREFIX = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized"
    config.DATASET = "Kinetics"
    config.LOSS_FILE = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_train_4_3_30.json"

    dataset_train = FrameSelectionLoader(
        cfg=config,
        pre_sampling_rate=4,
        selection_method="adaptive",
        num_frames=16,
        augmentations=True,
        return_type="Dict",
        mode = "train"
    )

    config.LOSS_FILE = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_val_4_3_30.json"

    dataset_val = FrameSelectionLoader(
        cfg=config,
        pre_sampling_rate=4,
        selection_method="adaptive",
        num_frames=16,
        augmentations=False,
        return_type="Dict",
        mode = "val"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="/graphics/scratch2/students/reutemann/timesformer_finetuning",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="/graphics/scratch2/students/reutemann/timesformer_finetuning_test/logs",
        logging_steps=500,
        evaluation_strategy="epoch",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("/home/reutemann/Dino-Video-Summarization-Transformer/timesformer_finetuning")

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

    # Save the plot as PNG file
    plt.savefig('eval_logs/finetuning_loss.png')


if __name__ == '__main__':
    finetuning()