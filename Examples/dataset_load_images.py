import torch
from datasets import load_dataset, Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ColorJitter, ToTensor

# load the beans dataset
dataset = load_dataset("beans", split="train")

# randomly change some of the images color properties
jitter = Compose(
    [ColorJitter(brightness=0.5, hue=0.5), ToTensor()]
)

# apply the transform to the dataset and generate the model input
def transforms(examples):
    examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
    return examples

# apply the data augmentations on-the-fly
dataset = dataset.with_transform(transforms)

# wrap the dataset and collate the samples into batches
def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["labels"])
        
    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)
