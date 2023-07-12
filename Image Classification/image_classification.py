from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
import numpy as np


# loading part of the food101 dataset from hugging face and splitting it into a train and test set

food = load_dataset("food101", split="train[:5000]")
food = food.train_test_split(test_size=0.2)

# print(food["train"][0])

# creating a dictionary that maps the label id to label name and the other way around

labels = food["train"].features["label"].names
label2id, id2label = dict(), dict()

for i, label in enumerate(labels):

    label2id[label] = str(i)
    id2label[str(i)] = label

# print(id2label[str(93)])

# load a ViT image processor to process the image into a tensor (data that is arranged in a multidimensional array)

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# applying some image transformations to the images to make it more robust against overfitting (model returns exact predictions for training data but not for new data)
# crop out a random part of the image, resize it and normalize ist with the image mean (sum of pixel values divided by pixel count) and standard deviation

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

# create a preprocessing function to apply the transforms and return the pixel values - the inputs to the model - of the image

def transforms(examples):

    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]

    return examples

# applying the preprocessing function aver the entire dataset. The transformations are applied on the fly when you load an element of the dataset

food = food.with_transform(transforms)

# creating a batch of examples - DefaultDataCollator does not apply additional preprocessing

data_collator = DefaultDataCollator()

# loading an evaluation method - in this case accuracy

accuracy = evaluate.load("accuracy")

# creating a function theat passes the predictions and labels to calculate the accuracy

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

# loading the model for image classification 

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# defining the training arguments

training_args = TrainingArguments(
    output_dir="image_classification_food_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# passing the arguments to the trainer with the model, dataset, tokenizer, data collator and compute_metrics function

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food["train"],
    eval_dataset=food["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

# train the model

trainer.train()

# saving the model locally

trainer.save_model("model")
