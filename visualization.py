import matplotlib.pyplot as plt
import json
import numpy as np
import torchvision.io as io


def plot_loss(loss_file_path, sampling_rate, key=None):
    # Load precomputed loss values
    with open(loss_file_path, 'r') as file:
        loss_dict = json.load(file)

    if key == None:
        _, loss_values = list(loss_dict.items())[0]
    
    else:
        loss_values = loss_dict[key]

    # loss_values = create_correlation_matrix(loss_values)

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
    plt.savefig("plots_test/test_msvd_finetuned_6.png", dpi=300)

    # free up memory
    plt.close()


def create_correlation_matrix(loss_list):
    correlation = np.corrcoef(loss_list, loss_list)
    normalized_correlation = correlation / correlation[len(loss_list) - 1]

    return normalized_correlation


def save_tensor_as_video(tensor):
    tensor = tensor.permute(1, 2, 3, 0)

    io.write_video('videos_test/video_2.mp4', tensor, fps=4)


if __name__ == '__main__':
    sampling_rate = 4
    loss_file_path = "loss_files_test/loss_output_test_msvd_finetuned_6.json"
    plot_loss(loss_file_path, sampling_rate)
    