import matplotlib.pyplot as plt
import json
import numpy as np


def plot_loss(file_path):
    # Load precomputed loss values (replace this with your actual data)
    with open(file_path, 'r') as file:
        loss_values = json.load(file)

    #frame_numbers = np.arange(len(loss_values["-aU-xCCzkT0_000061_000071"]))
    frame_numbers = np.arange(50)

    # Create a plot of loss values
    plt.bar(frame_numbers, loss_values["-aU-xCCzkT0_000061_000071"][:50], label='Loss', color='blue')
    plt.xlabel('Frame Number')
    plt.ylabel('Loss Value')
    plt.title('Loss Over Frames')
    plt.legend()
    plt.grid(False)

    # Save the plot
    plt.savefig("plots_test/test1.png")


if __name__ == '__main__':
    file_path = "loss_k400_resized_test/loss_output_fintuned_3_60.json"
    plot_loss(file_path)