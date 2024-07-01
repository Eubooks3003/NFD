import cv2
import numpy as np


import cv2
import numpy as np
import os

def segment_color_objects(image_path, output_path_red, output_path_green):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define thresholds for red and green
    lower_red1 = (0, 70, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 70, 50)
    upper_red2 = (180, 255, 255)
    lower_green = (40, 40, 40)
    upper_green = (80, 255, 255)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(output_path_red, mask_red)
    # cv2.imwrite(output_path_green, mask_green)

def process_folder(input_folder, output_folder_red, output_folder_green):
    if not os.path.exists(output_folder_red):
        os.makedirs(output_folder_red)
    if not os.path.exists(output_folder_green):
        os.makedirs(output_folder_green)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            output_path_red = os.path.join(output_folder_red, f'red_{filename}')
            output_path_green = os.path.join(output_folder_green, f'green_{filename}')
            segment_color_objects(file_path, output_path_red, output_path_green)
            print(f"Processed {filename}")

# Example usage
input_folder = '/home/ellina/Working/NFD/output_images_train'  # Folder with input images
output_folder_red = '/home/ellina/Working/NFD/output_masks_red'  # Folder to save red masks
output_folder_green = '/home/ellina/Working/NFD/output_masks_green'  # Folder to save green masks
process_folder(input_folder, output_folder_red, output_folder_green)

