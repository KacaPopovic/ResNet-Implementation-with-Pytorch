import os
import pandas as pd
from PIL import Image
from pathlib import Path

# Load the original CSV file
dataset_path = '/mnt/data/data.csv'
df = pd.read_csv('data.csv', sep=';')

# List to hold the rows for the new CSV
augmented_data = []

# Process each image in the dataset
for index, row in df.iterrows():
    original_path = row['filename']
    crack_label = row['crack']
    inactive_label = row['inactive']

    # Load the original image
    image = Image.open(original_path)

    # Flip the image horizontally
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Generate the new filename for the flipped image
    path_parts = os.path.split(original_path)
    flipped_filename = f"{path_parts[0]}/{Path(path_parts[1]).stem}_flipped.png"

    # Save the flipped image in the same directory as the original
    flipped_image.save(flipped_filename)

    # Add rows for both original and flipped images to the new CSV data
    augmented_data.append([original_path, crack_label, inactive_label])
    augmented_data.append([flipped_filename, crack_label, inactive_label])

# Create a DataFrame with the augmented data
augmented_df = pd.DataFrame(augmented_data, columns=['filename', 'crack', 'inactive'])

# Save the new DataFrame to a CSV file
augmented_csv_path = 'data_augmented.csv'
augmented_df.to_csv(augmented_csv_path, index=False, sep=';')

augmented_csv_path, augmented_df.head()