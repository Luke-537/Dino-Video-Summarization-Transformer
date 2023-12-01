import csv
import random
from collections import defaultdict

def extract_entries(input_file, output_file):
    # Read the input CSV file
    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = defaultdict(list)
        for row in reader:
            if row:  # Check if the row is not empty
                values = row[0].split()  # Split the row into two values
                if len(values) == 2:
                    data[values[1]].append(row[0])  # Append the whole row to the corresponding key

    # Process the data
    processed_data = []
    for key, values in data.items():
        if len(values) > 10:
            values = random.sample(values, 10)  # Randomly select 10 items if more than 10 are present
        processed_data.extend(values)

    # Write the output CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for line in processed_data:
            writer.writerow([line])

if __name__ == '__main__':
    extract_entries(
        '/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations/test.csv', 
        '/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations/test_small.csv'
    )