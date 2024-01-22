import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import evaluate
import pickle

from pycocoevalcap.cider.cider import Cider

def main(selection_method="adaptive", video_tag=None):
    """
    Compute the captioning metrics BLEU, METEOR, BERT and CIDEr for the test split of MSVD 
    with generated captions from Video-LLaVA.

    selection_method (String): "adaptive" or "uniform".
    video_tag (String): metrics for a specific video.
    """
    
    if selection_method == "uniform":
        path = "/home/reutemann/Dino-Video-Summarization-Transformer/eval_logs/captions_uniform.csv"
    else:
        path = "/home/reutemann/Dino-Video-Summarization-Transformer/eval_logs/captions_adaptive.csv"
    captions_dict = {}

    # read the file containing captioins
    with open(path, 'r') as file:
        for line in file:
            # split each line by space and remove the last element if it is '</s>'
            parts = line.strip().split(' ')
            if parts[-1] == '</s>"':
                parts = parts[:-1]  # Remove the '</s>' tag
            key = parts[0]
            key = key[:-4]
            value = ' '.join(parts[1:]).replace('"', '')  # combine the remaining parts and remove quotes
            value = value.replace('</s>', '').strip()  # remove '</s>' from the value
            captions_dict[key] = [value]
    
    # sorting by the video keys
    sorted_keys = sorted(captions_dict.keys())
    captions_ordered = {key: captions_dict[key] for key in sorted_keys}

    # extract singular tag if needed
    if video_tag is not None:
        captions_ordered = {video_tag : captions_ordered[video_tag]}

    truth_dict = {}

    # read the file containing ground-truth
    with open("/home/reutemann/Dino-Video-Summarization-Transformer/eval_logs/annotations.csv", 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            key = parts[0]
            value = ' '.join(parts[1:])

            if key in truth_dict:
                # append the new caption to the existing list for this key
                truth_dict[key].append(value)
            else:
                # create a new list for this key
                truth_dict[key] = [value]

    # sorting by the video keys
    sorted_keys = sorted(truth_dict.keys())
    truth_ordered = {key: truth_dict[key] for key in sorted_keys}
    
    # extract singular tag if needed
    if video_tag is not None:
        truth_ordered = {video_tag : truth_ordered[video_tag]}

    # check if the keys of the dictionaries match
    keys_match_in_order = list(captions_ordered.keys()) == list(truth_ordered.keys())

    if keys_match_in_order:
        print("All keys match")
    else:
        print("Keys do not match")

    # extract only the captions for the test split
    test_path = '/graphics/scratch/datasets/MSVD/MSVD_video_with_caption_test.pkl'

    with open(test_path, 'rb') as file:

        data = pickle.load(file)

    test_set = list(sorted(set(data['video_name'])))

    predictions = []
    references = []

    # create lists of predictions and references
    if video_tag is not None:
        predictions.append(captions_ordered[video_tag][0])
        references.append(truth_ordered[video_tag])
    else:
        for i in range(len(test_set)):
            predictions.append(captions_ordered[test_set[i]][0])
            references.append(truth_ordered[test_set[i]])
    
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

    print("BLEU", bleu_score)
    print("METEOR", meteor_score)
    print("BERT", bert_score)
    print("CIDEr", cider_score)


if __name__== "__main__":
    main("uniform", video_tag="rOic25PnIx8_1_3")
    main("adaptve", video_tag="rOic25PnIx8_1_3")
