import os
import csv


def main():
    """
    Creating a CSV file containing the file names from the MSVD videos
    """

    directory_path = '/graphics/scratch/datasets/MSVD/YouTubeClips'
    file_names = os.listdir(directory_path)
    file_names = [f for f in file_names if os.path.isfile(os.path.join(directory_path, f))]

    with open('test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for filename in file_names:
            writer.writerow([filename + " 0"])  

if __name__ == '__main__':
    main()