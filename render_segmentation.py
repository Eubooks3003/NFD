import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def load_pkl_file(filepath):
    """Load a pickle file and return its content."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_masks_from_cameras(masks, output_folder, index, num_masks_per_camera=4):
    """Save a specified number of masks from each camera to a folder."""
    num_cameras = masks.shape[1]
    num_masks = masks.shape[0]
    
    if num_masks_per_camera > num_masks:
        print("Requested more masks per camera than available. Saving available masks only.")
        num_masks_per_camera = num_masks
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for camera_index in range(num_cameras):
        for mask_index in range(num_masks):
            mask = masks[mask_index, camera_index, :, :]
            filename = f'Camera_{camera_index + 1}_Mask_{mask_index + 1}_Index_{index + 1}.png'
            filepath = os.path.join(output_folder, filename)
            plt.figure()
            plt.imshow(mask, cmap='tab20')  # Use a colormap that clearly distinguishes mask labels
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
            masks = load_pkl_file(file_path)
            print(f"Loaded {len(masks)} masks from {file}.")
            print("Masks Size: ", masks.shape)
            save_masks_from_cameras(masks, output_folder, index)
        
        index += 1

# Specify the input and output directories
input_folder = '/home/ellina/Working/NFD/sweeping-piles-train/segm'  # Folder containing .pkl files with masks
output_folder = '/home/ellina/Working/NFD/output_masks'             # Folder to save processed masks

# Process all .pkl files in the specified folder
process_folder(input_folder, output_folder)
