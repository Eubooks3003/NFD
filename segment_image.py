import cv2
import numpy as np


import cv2
import numpy as np
import os

def setup_and_clean_directory(directory):
    """Checks if the directory exists and creates it if not, then clears any existing files."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def fill_goal_region(mask_green):
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask_green, False  # Return the original mask if no contours are found

    # Create an empty mask to draw the filled contour
    filled_mask = np.zeros_like(mask_green)

    # Assume the largest contour is the square border and fill it
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(filled_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)

    return filled_mask, True



def segment_color_objects(image_path, output_path_red, output_path_green):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    print(image)
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

    if np.sum(mask_green) == 0:
        print("The green mask is empty.")
    else:
        filled_mask, success = fill_goal_region(mask_green)
        if success:
            cv2.imwrite(output_path_green, filled_mask)


    cv2.imwrite(output_path_red, mask_red)

def process_folder(input_folder, output_folder_red, output_folder_green):
    setup_and_clean_directory(output_folder_red)
    setup_and_clean_directory(output_folder_green)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            output_path_red = os.path.join(output_folder_red, f'{filename}')
            output_path_green = os.path.join(output_folder_green, f'green_{filename}')
            segment_color_objects(file_path, output_path_red, output_path_green)
            print(f"Processed {filename}")

# Example usage
input_folder = '/home/ellina/Working/NFD/output_images_test'  # Folder with input images
output_folder_red = '/home/ellina/Working/NFD/output_masks_red'  # Folder to save red masks
output_folder_green = '/home/ellina/Working/NFD/output_masks_green'  # Folder to save green masks
process_folder(input_folder, output_folder_red, output_folder_green)

