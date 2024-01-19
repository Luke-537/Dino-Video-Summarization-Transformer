import evaluate
import pickle
from pycocoevalcap.cider.cider import Cider

def main(selection_method="adaptive"):
    
    if selection_method == "uniform":
        path = "/home/reutemann/Dino-Video-Summarization-Transformer/eval_logs/captions_uniform.csv"
    else:
        path = "/home/reutemann/Dino-Video-Summarization-Transformer/eval_logs/captions_adaptive.csv"
    captions_dict = {}
    # Read the file line by line
    with open(path, 'r') as file:
        for line in file:
            # Split each line by space and remove the last element if it is '</s>'
            parts = line.strip().split(' ')
            if parts[-1] == '</s>"':
                parts = parts[:-1]  # Remove the '</s>' tag
            key = parts[0]
            key = key[:-4]
            value = ' '.join(parts[1:]).replace('"', '')  # Combine the remaining parts and remove quotes
            value = value.replace('</s>', '').strip()  # Remove '</s>' from the value
            captions_dict[key] = [value]

    sorted_keys = sorted(captions_dict.keys())
    captions_ordered = {key: captions_dict[key] for key in sorted_keys}

    captions_ordered = {"giLxPCgLLqg_9_19" : captions_ordered["giLxPCgLLqg_9_19"]}

    truth_dict = {}

    with open("/home/reutemann/Dino-Video-Summarization-Transformer/eval_logs/annotations.csv", 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            key = parts[0]
            value = ' '.join(parts[1:])

            if key in truth_dict:
                # Append the new caption to the existing list for this key
                truth_dict[key].append(value)
            else:
                # Create a new list for this key
                truth_dict[key] = [value]

    sorted_keys = sorted(truth_dict.keys())
    truth_ordered = {key: truth_dict[key] for key in sorted_keys}

    truth_ordered = {"giLxPCgLLqg_9_19" : truth_ordered["giLxPCgLLqg_9_19"]}

    keys_match_in_order = list(captions_ordered.keys()) == list(truth_ordered.keys())

    if keys_match_in_order:
        print("All keys match")
    else:
        print("Keys do not match")

    test_path = '/graphics/scratch/datasets/MSVD/MSVD_video_with_caption_test.pkl'

    # Open the file in binary read mode
    with open(test_path, 'rb') as file:
        # Load the object from the file
        data = pickle.load(file)

    test_set = list(sorted(set(data['video_name'])))

    predictions = []
    references = []

    #for i in range(len(test_set)):
    #    predictions.append(captions_ordered[test_set[i]][0])
    #    references.append(truth_ordered[test_set[i]])

    predictions.append(captions_ordered["giLxPCgLLqg_9_19"][0])
    references.append(truth_ordered["giLxPCgLLqg_9_19"])
    
    # calculate BERT score
    bert = evaluate.load("bertscore")
    bert_results = bert.compute(predictions=predictions, references=references, lang = "en")
    bert_score = sum(bert_results["precision"])/len(bert_results["precision"])

    # calculate CIDEr score
    test_captions = {key: captions_dict[key] for key in test_set}
    test_truth = {key: truth_dict[key] for key in test_set}
    scorer = Cider()
    cider_score, _ = scorer.compute_score(test_truth, test_captions)

    # calculate BLEU Score
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)

    # calculate METEOR Score
    bleu = evaluate.load("meteor")
    meteor_score = bleu.compute(predictions=predictions, references=references)

    print("BLEU", bleu_score, )
    print("METEOR", meteor_score)
    print("BERT", bert_score)
    print("CIDEr", cider_score)


if __name__== "__main__":
    main("uniform")
    main("adaptve")
