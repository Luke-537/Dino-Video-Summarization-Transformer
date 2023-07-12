from datasets import load_dataset
from transformers import pipeline


ds = load_dataset("food101", split="validation[:10]")
image = ds["image"][0]

classifier = pipeline("image-classification", model="Luke537/image_classification_food_model")
print(classifier(image))
