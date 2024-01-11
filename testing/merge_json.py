import json


def merge_json(path_1, path_2, path_new):

    with open(path_1, 'r') as file:
        dict1 = json.load(file)

    with open(path_2, 'r') as file:
        dict2 = json.load(file)

    merged_dict = dict1.copy()  
    merged_dict.update(dict2) 

    with open(path_new, 'w') as file:
        json.dump(merged_dict, file, indent=4)

if __name__ == '__main__':
    path_1 = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_train-5_4_3_30.json"
    path_2 = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_train-6_4_3_30.json"
    path_new = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_train_4_3_30.json"
    merge_json(path_1, path_2, path_new)