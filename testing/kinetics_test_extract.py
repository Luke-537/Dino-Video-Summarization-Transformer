import csv
import random
from collections import defaultdict

def extract_entries(input_file, output_file, item_count):
    """
    Extracting a subset of a dataset that is provided as a CSV file.

    input_file (String): path to source file.
    input_file (String): path to target file.
    item_count (int): number of items to be extracted for each class.
    """

    # reading the input CSV file
    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = defaultdict(list)
        for row in reader:
            if row:
                values = row[0].split()
                if len(values) == 2:
                    data[values[1]].append(row[0])

    # process the data
    processed_data = []
    for key, values in data.items():
        if len(values) > item_count:
            values = random.sample(values, item_count)  # randomly select 15 items if more than 15 are present
        processed_data.extend(values)

    # write the output CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for line in processed_data:
            writer.writerow([line])

if __name__ == '__main__':
    item_count = 15
    extract_entries(
        '/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations/test.csv', 
        '/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations/test_small.csv',
        item_count
    )