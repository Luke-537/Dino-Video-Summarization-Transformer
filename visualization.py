import matplotlib.pyplot as plt
import json
import numpy as np
from loss_utils import create_correllation_matrix

def plot_loss(file_path):
    # Load precomputed loss values (replace this with your actual data)
    with open(file_path, 'r') as file:
        loss_values = json.load(file)

    loss_values = loss_values["-4wsuPCjDBc_5_15"]

    frame_numbers = np.arange(len(loss_values))

    # Create a figure and a set of subplots. This utility wrapper makes it convenient to create common layouts of subplots.
    # Adjusting the figsize parameter will give you a wider plot.
    # For instance, here it's set to 12 inches wide and 6 inches tall.
    fig, ax = plt.subplots(figsize=(20, 6))

    # Create a bar chart. The ax.bar function is preferable over plt.bar when dealing with subplots.
    ax.bar(frame_numbers, loss_values, label='Loss', color='blue')

    # Set labels and title
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Over Frames')
    ax.legend()

    # You can choose whether to display the grid.
    ax.grid(False)

    # Save the plot with higher resolution (dpi).
    # You can increase the dpi value further if you need more detail.
    plt.savefig("plots_test/test_msvd_5.png", dpi=300)

    # It's good practice to close the figure with plt.close() to free up memory, especially if running in a loop.
    plt.close()

if __name__ == '__main__':
    file_path = "loss_files_test/loss_output_test_msvd_4.json"
    plot_loss(file_path)
    