from transformers import pipeline

# creating a pipeline instance and specitying its task
classifier = pipeline("sentiment-analysis")

# use the classifier for the target text
results = classifier(["I am happy to use the 🤗 Transformers library.", "I am kinda not really sad."])

# passing a list of inputs to the pipeline to return a list of dictionaries
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
