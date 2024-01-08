import evaluate

captions_dict = {}

predictions = ["hello there general kenobi", "foo bar foobar"]

# Read the file line by line
with open("/home/reutemann/Dino-Video-Summarization-Transformer/eval_logs/captions_adaptive.csv", 'r') as file:
    for line in file:
        # Split each line by space and remove the last element if it is '</s>'
        parts = line.strip().split(' ')
        if parts[-1] == '</s>"':
            parts = parts[:-1]  # Remove the '</s>' tag
        key = parts[0]
        key = key[:-4]
        value = ' '.join(parts[1:]).replace('"', '')  # Combine the remaining parts and remove quotes
        value = value.replace('</s>', '').strip()  # Remove '</s>' from the value
        captions_dict[key] = value

sorted_keys = sorted(captions_dict.keys())
captions_ordered = {key: captions_dict[key] for key in sorted_keys}

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

keys_match_in_order = list(captions_ordered.keys()) == list(truth_ordered.keys())

print(keys_match_in_order)

predictions = list(captions_ordered.values())
references = list(truth_ordered.values())

breakpoint()

bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)