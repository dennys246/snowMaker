import os, csv
from glob import glob


"""
This script is used for taking labels assessed from label segments
of the snowpack data and labeling the rest of the datasets segmented files.
"""

# Define runtime parameters
dataset_folder = "/Users/dennyschaedig/Datasets/rocky_mountain_snowpack/"

label_folder = "labels/mini-core_labels/"
raw_folder = "raw/"
segment_folders = [
    'segmented/cores/',
    'segmented/magnified-profiles/',
    'segmented/profiles/'
]

extract_number = lambda filename : int(filename.split('IMG_')[-1].split('_')[0].split('.JPG')[0])

# Grab all label files
label_files = glob(f"{dataset_folder}{label_folder}/*")

# Grab the label photo numbers
label_filenumbers = [os.path.basename(file) for file in label_files] # Remove the image paths and leave just the filenames

# Sort files in ascending order
filenumbers = [extract_number(file) for file in label_filenumbers]
zipper = sorted(zip(filenumbers, label_files))
filenumbers, label_files = zip(*zipper)

crystal_labels = []

# Iterate through all segmented folder
for segment_folder in segment_folders:
    segment_images = glob(f"{dataset_folder}{segment_folder}*") # Grab all segmented images

    # Grab the label photo numbers
    segment_filenumbers = [os.path.basename(file) for file in segment_images] # Remove the image paths and leave just the filenames

    # Sort files in ascending order
    filenumbers = [extract_number(file) for file in segment_filenumbers]
    zipper = sorted(zip(segment_filenumbers, segment_images))
    filenumbers, segment_images = zip(*zipper)

    # Iterate through each label file
    for label_ind, label_file in enumerate(label_files):

        # Grab the label for the image
        label = '_'.join([label for label in label_file.split('.JPG')[0].split('_')[-4:] if len(label) == 1]).split('.')[0]
        label_image_number = extract_number(label_file)

        if len(label_files) == label_ind + 1: # If there is no next image
            next_label_file = "IMG_999999_0_0_0_0.JPG" # Set next image as "infinity"
            next_image_number = 999999  # large dummy value
        
        else: # Grab next label
            next_label_file = label_files[label_ind + 1]
            next_image_number = extract_number(next_label_file)
        
        # Iterate through all raw files
        for image_file in segment_images:
            image_number = extract_number(image_file) 
            # If the file is between the current and next label
            if image_number >= label_image_number and image_number < next_image_number:
                path = os.path.dirname(image_file)
                filename = os.path.basename(image_file)
                new_name = f"IMG_{image_number}_{label}.JPG" # Rename with label
                
            if image_number >= next_image_number:
                break