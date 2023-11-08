import matplotlib.pyplot as plt
import json
import numpy as np
import torchvision.io as io
import os
import torch

def plot_loss(loss_file_path, sampling_rate, key=None):
    # Load precomputed loss values
    with open(loss_file_path, 'r') as file:
        loss_dict = json.load(file)

    if key == None:
        _, loss_values = list(loss_dict.items())[0]
        key = "no_key"
    
    else:
        loss_values = loss_dict[key]

    frame_numbers = np.arange(len(loss_values))

    # Create a figure and a set of subplots
    # Fihure size set to 12 inches wide and 6 inches tall
    fig, ax = plt.subplots(figsize=(15, 6))

    # Create a bar chart
    ax.bar(frame_numbers*sampling_rate, loss_values, label='Loss', color='blue', width=2.5)

    # Set labels and title
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Over Frames')
    ax.legend()
    ax.grid(False)

    # Save the plot with higher resolution (dpi).
    plt.savefig('plots_test/' + key, dpi=300)

    # free up memory
    plt.close()


def plot_matrix(loss_file_path, key=None):
        # Load precomputed loss values
    with open(loss_file_path, 'r') as file:
        loss_dict = json.load(file)

    if key == None:
        _, loss_values = list(loss_dict.items())[0]
        key = "no_key"
    
    else:
        loss_values = loss_dict[key]

    loss_values = create_correlation_matrix(loss_values)

    frame_numbers = np.arange(len(loss_values))

    # Create a figure and a set of subplots
    # Fihure size set to 12 inches wide and 6 inches tall
    fig, ax = plt.subplots(figsize=(15, 6))

    # Create a bar chart
    ax.bar(frame_numbers, loss_values, label='Loss', color='blue', width=2.5)

    # Set labels and title
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Over Frames')
    ax.legend()
    ax.grid(False)

    # Save the plot with higher resolution (dpi).
    plt.savefig('plots_test/' + key + '_matrix', dpi=300)

    # free up memory
    plt.close()


def create_correlation_matrix(loss_list):
    loss_tensor = torch.tensor([loss_list, loss_list], dtype=torch.float32)

    matrix = torch.corrcoef(loss_tensor)

    breakpoint()

    return matrix


def save_tensor_as_video(tensor, video_path):
    tensor = tensor.permute(1, 2, 3, 0)
    video_name = os.path.basename(video_path)
    video_name_without_extension, extension = os.path.splitext(video_name)

    io.write_video('videos_test/' + video_name_without_extension + '.mp4', tensor, fps=8)


if __name__ == '__main__':
    sampling_rate = 4
    loss_file_path = "loss_files_test/loss_msvd_no_finetuning_1.json"
    #plot_loss(loss_file_path, sampling_rate, "-8y1Q0rA3n8_108_115")
    plot_matrix(loss_file_path, "-4wsuPCjDBc_5_15")
    