import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_pkl_file(filepath):
    """Load a pickle file and return its content."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_images_from_cameras(images, output_folder, index, num_images_per_camera=4):
    """Save a specified number of images from each camera to a folder."""
    num_cameras = images.shape[1]
    num_images = images.shape[0]
    
    if num_images_per_camera > num_images:
        print("Requested more images per camera than available. Saving available images only.")
        num_images_per_camera = num_images
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Camera Index is which Camera
    # Image Index is which image in the sequence 
    # Index is which sequence
    for camera_index in range(num_cameras):
        for image_index in range(num_images):
            image = images[image_index, camera_index, :, :, :]
            filename = f'Camera_{camera_index + 1}_Index_{index + 1}_Image_{image_index + 1}.png'
            filepath = os.path.join(output_folder, filename)
            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved: {filepath}")

def process_folder(input_folder, output_folder):
    """Process all .pkl files in the specified folder."""
    index = 0
    for file in os.listdir(input_folder):
        if file.endswith('.pkl'):
            file_path = os.path.join(input_folder, file)
            images = load_pkl_file(file_path)
            print(f"Loaded {len(images)} images from {file}.")
            print("Images Size: ", images.shape)
            save_images_from_cameras(images, output_folder, index)
        
        index += 1

# Specify the input and output directories
input_folder = '/home/ellina/Working/NFD/sweeping-piles-test/color'  # Folder containing .pkl files
output_folder = '/home/ellina/Working/NFD/output_images_train'             # Folder to save processed images

# Process all .pkl files in the specified folder
process_folder(input_folder, output_folder)
