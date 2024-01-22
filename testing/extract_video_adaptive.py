import sys
sys.path.append('/home/reutemann/Dino-Video-Summarization-Transformer')
import os
import torch
import torch.utils.data
import torchvision.io as io
import json
import numpy as np

from testing.visualization import save_tensor_as_video, plot_loss
from torchvision.utils import save_image


def extract_video(video_path, loss_path, pre_sampling_rate, selection_method, out_path, save_frames=False):
    """
    Extracting the adaptively and uniformly sampled frames, saving them as video and creating the loss plot.
    Optionally also saving the individual frames.

    video_path (String): path to the video.
    loss_path (String): path to the file contaiing the loss values.
    pre-sampling_rate (int): pre-sampling rate to downsample the video.
    selection_method (String): "uniform" or "adaptive".
    out_path (String): path to the output directory.
    save_frames (Bool): whether to extract the indivual frames or not.
    """
     
    video, _, _ = io.read_video(video_path, pts_unit='sec')
    frames_unsampled = video.to(torch.float)

    # open JSON file containing loss values
    with open(loss_path, 'r') as file:
        loss_dict = json.load(file)

    frames_sampled = frames_unsampled[::pre_sampling_rate]

    # T H W C -> T C H W.
    frames = frames_sampled.permute(0, 3, 1, 2)

    N = 8  # number of frames to select

    file_name = os.path.basename(video_path)

    if selection_method == "adaptive":   
        # get the file name and then the loss values
        key = os.path.splitext(file_name)[0]
        loss_list = loss_dict[key]

        # optionally sharpening the values
        #loss_list = np.asarray(loss_list) ** 2
        loss_list = np.asarray(loss_list)

        if len(loss_list) > frames.size(0):
            loss_list = loss_list[:frames.size(0)]

        # min-max normalization
        pdf = (loss_list - loss_list.min()) / (loss_list.max() - loss_list.min())

        # normalizing the loss values to create a PDF, might need to scale
        pdf = loss_list / np.sum(loss_list)

        # creating the CDF from the PDF
        cdf = np.cumsum(pdf)

        selected_frames = []
        indices = []
        for i in range(N):
            # finding the frame index corresponding to the quantile
            j = i / N
            cdf_array = np.asarray(cdf)
            idx = (np.abs(cdf_array - j)).argmin()

            # scaling the index to the unsampled video length
            idx_scaled = idx*pre_sampling_rate

            if idx_scaled not in indices:
                indices.append(idx_scaled)
                selected_frames.append(frames[idx])

            # if the index has already been selected, try again for the next possible index until the end of the video
            else:
                temp = idx_scaled + pre_sampling_rate
                search = True
                while search:
                    if temp not in indices and temp < frames_unsampled.size(0):
                        indices.append(temp)
                        selected_frames.append(frames[idx])
                        search = False

                    elif temp >= frames_unsampled.size(0):
                        indices.append(temp - pre_sampling_rate)
                        selected_frames.append(frames[-1])
                        search = False

                    else:
                        temp = temp + pre_sampling_rate

        # creating a tensor from the list of frames
        frames = torch.stack(selected_frames)
        
    else:
        # sampling N frames uniformly
        selected_frames = []
        interval = int(frames.size(0) / N)

        for i in range(N):
            selected_frames.append(frames[i*interval])

        frames = torch.stack(selected_frames)
        indices = None

    # T C H W -> C T H W.
    frames = frames.permute(1, 0, 2, 3)
    
    # saving the resulting tensor
    save_tensor_as_video(frames, out_path)
    
    # optionally saving the individual frames of the video as PNGs
    if save_frames:
        key = os.path.splitext(file_name)[0]
        frames = frames.permute(1, 0, 2, 3) / 255

        for i in range(len(frames)):
            path = 'videos_sampled/'+ str(key) + '/' + str(i) + '.png'
            save_image(frames[:][i], path)

    # returning the list of selected indices
    return indices

if __name__ == '__main__':
    # video key and selecting the dataset
    key = "SZP3Jpbbwj0_52_59"
    if True:
        loss_path = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_msvd_4_3_30.json"
        video_path = "/graphics/scratch/datasets/MSVD/YouTubeClips/" + key + ".avi"
    else:
        loss_path = "/home/reutemann/Dino-Video-Summarization-Transformer/loss_values_new/loss_kinetics_test_4_3_30.json"
        video_path = "/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/test/" + key + ".mp4"

    # defining the output path
    dir_path = "/home/reutemann/Dino-Video-Summarization-Transformer/videos_sampled/" + key
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # extracting the video uniformly
    out_path = dir_path + "/" + key + "_u.mp4"
    _ = extract_video(video_path, loss_path, 4, "uniform", out_path, save_frames=False)

    # extracting the video adaptively
    out_path = dir_path + "/" + key + "_a.mp4"
    indices = extract_video(video_path, loss_path, 4, "adaptive", out_path, save_frames=True)

    # plotting the loss values
    plot_loss(loss_path, 4, dir_path + "/" + key + ".png", key=key, selected_frames=indices)