import matplotlib.pyplot as plt
import json
import numpy as np
import torchvision.io as io


def plot_loss(loss_file_path, sampling_rate, plot_path, key=None, selected_frames=None):
    """
    Creating and exporting a plot for a video with its loss values.

    loss_file_path (String): path to the file containing loss values.
    sampling_rate (int): pre-sampling rate to consider in the plot.
    key (String or int): video key or index in loss file.
    selected_frames (List[int]): list containing selected indices.
    """
        
    # load precomputed loss values
    with open(loss_file_path, 'r') as file:
        loss_dict = json.load(file)

    # extracting the loss values
    if isinstance(key, str):
        loss_values = loss_dict[key]
    elif isinstance(key, int):
        _, loss_values = list(loss_dict.items())[key]
    else:
        _, loss_values = list(loss_dict.items())[0]

    # creating list [0, 1, 2, ...] of length equal to the size of loss values
    frame_numbers = np.arange(len(loss_values))

    # creating a figure and a set of subplots, figure size set to 15 inches wide and 6 inches tall
    fig, ax = plt.subplots(figsize=(15, 6))

    # creating a bar chart
    ax.plot(frame_numbers*sampling_rate, loss_values, label='Loss', color='steelblue', linewidth=2.5)

    # displaying the selected frames if provided
    if selected_frames is not None:
        selected_loss_values = [loss_values[int(i/sampling_rate)] for i in selected_frames]
        ax.scatter(selected_frames, selected_loss_values, label='Selected Frames', color='crimson', zorder=5, s=70)


    # setting labels and title
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Over Frames')
    ax.legend()
    ax.grid(False)

    plt.savefig(plot_path, dpi=300)
    plt.close()

def save_tensor_as_video(tensor, video_path):
    """
    Writing a tensor as an MP4 to a directory.
    """

    tensor = tensor.permute(1, 2, 3, 0)
    io.write_video(video_path, tensor, fps=8)


if __name__ == '__main__':
    # defining video key, path to loss file and output path
    key = "SZP3Jpbbwj0_52_59"
    sampling_rate = 4
    loss_file_path = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values/loss_msvd_4_3_30.json"
    export_path = "/home/reutemann/Dino-Video-Summarization-Transformer/test_data/loss_msvd_4_3_30.png"

    plot_loss(loss_file_path, sampling_rate, export_path, key)
    