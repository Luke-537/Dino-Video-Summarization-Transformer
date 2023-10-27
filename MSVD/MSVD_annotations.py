import os
import csv

directory_path = '/graphics/scratch/datasets/MSVD/YouTubeClips'  

file_names = os.listdir(directory_path)  

file_names = [f for f in file_names if os.path.isfile(os.path.join(directory_path, f))]

with open('annotations.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for filename in file_names:
        writer.writerow([filename + " 0"])  
