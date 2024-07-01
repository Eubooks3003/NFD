import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2  # Import OpenCV for handling image operations
import re

def load_pkl_file(filepath):
    """Load and return the contents of a pickle file."""
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def get_image_size(image_folder):
    """Get the size of the first image in the folder to set the plot dimensions."""
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_folder, file)
            image = cv2.imread(path)
            if image is not None:
                return image.shape[:2]  # Returns (height, width)
    raise RuntimeError("No valid images found in the folder.")

def draw_rectangle(ax, center, length, width, angle):
    """Draw a rectangle centered at a given point with a given rotation."""
    rect = patches.Rectangle((center[0] - width / 2, center[1] - length / 2), width, length, angle=angle, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

def add_centered_rotated_rectangle(ax, center, width, height, angle):
    """
    Add a centered and rotated rectangle to an axes.

    Args:
    ax: The axes object where the rectangle will be added.
    center: Tuple (x, y) representing the center of the rectangle.
    width: The width of the rectangle.
    height: The height of the rectangle.
    angle: The rotation angle of the rectangle in degrees.
    """
    # Angle in radians
    angle_rad = np.radians(-angle)

    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rectangle corners relative to the center
    half_width, half_height = width / 2, height / 2
    corners = np.array([
        [-half_width, -half_height],  # Bottom left
        [half_width, -half_height],   # Bottom right
        [half_width, half_height],    # Top right
        [-half_width, half_height]    # Top left
    ])

    # Rotate and offset corners
    rotated_corners = np.dot(corners, R) + center

    # Create polygon and add to axes
    rectangle = patches.Polygon(rotated_corners, closed=True, edgecolor='none', facecolor='w', linewidth=1)  # Fill color set to white
    ax.add_patch(rectangle)
def plot_trajectory(action, image_index, output_folder, image_size, extend_factor, index):
    """Plot extended 2D trajectory with individual rectangle images."""
    if action is None:
        print("No action provided for plot.")
        return

    fig, ax = plt.subplots(figsize=(image_size[1] / 100, image_size[0] / 100))
    pose0_position = action['pose0'][0]
    pose1_position = action['pose1'][0]

    transform_x = lambda x: (x - 0.5) * image_size[1] + image_size[1] / 2
    transform_y = lambda y: y * image_size[0] + image_size[0] / 2

    original_xs = [transform_x(pose0_position[0]), transform_x(pose1_position[0])]
    original_ys = [transform_y(pose0_position[1]), transform_y(pose1_position[1])]

    direction = np.array([original_xs[1] - original_xs[0], original_ys[1] - original_ys[0]])
    norm_direction = direction / np.linalg.norm(direction)
    extended_start = np.array([original_xs[0], original_ys[0]]) - norm_direction * extend_factor * np.linalg.norm(direction)
    extended_end = np.array([original_xs[1], original_ys[1]]) + norm_direction * extend_factor * np.linalg.norm(direction)

    # ax.plot([extended_start[0], extended_end[0]], [extended_start[1], extended_end[1]], 'go-', label= f'Camera_1_Index_{index + 1}_Image_{image_index + 1}.png')
    positions = [extended_start, extended_end]
    angles = [np.degrees(np.arctan2(direction[1], direction[0]) + np.pi / 2) for _ in positions]
    rect_width, rect_height = 40, 20

    for pos in [extended_start, extended_end]:
        rectangle_angle = np.degrees(np.arctan2(direction[1], direction[0]) + np.pi / 2)
        rect_width, rect_height = 40, 20  # Example sizes for rectangles
        add_centered_rotated_rectangle(ax, (pos[0], pos[1]), rect_width, rect_height, rectangle_angle)

    # Save the main plot
    # main_output_path = os.path.join(output_folder, f'extended_action_{index}.png')
    # plt.savefig(main_output_path, format='png', dpi=300)
    plt.close(fig)

    # Create individual plots for each rectangle
    for i, pos in enumerate(positions):
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black') 
        # ax.plot([extended_start[0], extended_end[0]], [extended_start[1], extended_end[1]], 'go-')
        add_centered_rotated_rectangle(ax, (pos[0], pos[1]), rect_width, rect_height, angles[i])
        ax.set_xlim([0, image_size[1]])
        ax.set_ylim([0, image_size[0]])
        ax.axis('off')
        rectangle_output_path = os.path.join(output_folder, f'Camera_1_Index_{index + 1}_Image_{image_index + 1}_Rectange_{i}.png')
        plt.savefig(rectangle_output_path, format='png', dpi=300)
        plt.close(fig)

    print(f"All plots saved: main plot and individual rectangles for Action {image_index} from Trajectory {index}")

def process_actions(filepath, output_folder, reference_image_folder, extend_factor):
    """Load an action file and plot each trajectory on an image-sized canvas."""
    index = 0
    actions = load_pkl_file(filepath)
    image_size = get_image_size(reference_image_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, action in enumerate(actions):
        plot_trajectory(action, i, output_folder, image_size, extend_factor, index)


def get_images(image_folder):
    """Retrieve a list of images from the specified folder that match 'Camera_1_Index_1_Image_j.png' and sort them by 'j'."""
    pattern = re.compile(r'Camera_1_Index_1_Image_(\d+)\.png', re.IGNORECASE)
    images = []
    for file in os.listdir(image_folder):
        match = pattern.match(file)
        if match:
            # Extract the integer 'j' for sorting
            index = int(match.group(1))
            path = os.path.join(image_folder, file)
            images.append((index, path))

    # Sort images based on the numeric 'j' extracted from filenames
    images.sort(key=lambda x: x[0])
    
    # Return only the paths, sorted by 'j'
    return [image[1] for image in images]

def overlay_trajectory_on_image(image_path, action, output_path, extend_factor):
    """Overlay extended trajectory on an image, draw rectangles at the new endpoints, and save the result."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image:", image_path)
        return

    if action is None:
        print("No action provided for overlay.")
        return

    # Define the color for the trajectory and rectangles (BGR format)
    line_color = (0, 255, 0)  # Green for line
    rect_color = (255, 0, 0)  # Red for rectangles
    thickness = 2  # Line and rectangle thickness

    height, width, _ = image.shape

    # Transform coordinates to match image dimensions centered at (0.5, 0)
    transform_x = lambda x: int((x - 0.5) * width + width / 2)
    transform_y = lambda y: int(y * height + height / 2)

    # Original start and end points
    pose0_position = action['pose0'][0]
    pose1_position = action['pose1'][0]
    start = (transform_x(pose0_position[0]), transform_y(pose0_position[1]))
    end = (transform_x(pose1_position[0]), transform_y(pose1_position[1]))

    # Calculate direction for line extrapolation
    direction = np.array([end[0] - start[0], end[1] - start[1]])
    norm_direction = direction / np.linalg.norm(direction)

    # Extend the line by a fixed percentage of its original length
    extended_start = np.array(start) - norm_direction * extend_factor * np.linalg.norm(direction)
    extended_end = np.array(end) + norm_direction * extend_factor * np.linalg.norm(direction)

    # Convert to integer for drawing
    extended_start = tuple(extended_start.astype(int))
    extended_end = tuple(extended_end.astype(int))

    # Draw the extended trajectory line
    cv2.line(image, extended_start, extended_end, line_color, thickness)

    # Draw rectangles at the new endpoints
    for pos in [extended_start, extended_end]:
        rect_width, rect_height = 40, 20  # Example sizes for rectangles
        top_left = (pos[0] - rect_width // 2, pos[1] - rect_height // 2)
        bottom_right = (pos[0] + rect_width // 2, pos[1] + rect_height // 2)
        cv2.rectangle(image, top_left, bottom_right, rect_color, thickness)

    # Save the image with the trajectory and rectangles
    cv2.imwrite(output_path, image)
    print(f"Overlay saved to {output_path}")

def process_actions_with_images(actions_file, images_folder, output_folder, extend_factor):
    """Process actions and overlay each on corresponding images."""
    actions = load_pkl_file(actions_file)
    image_paths = get_images(images_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each action and corresponding image
    for i, (action, image_path) in enumerate(zip(actions, image_paths)):
        output_path = os.path.join(output_folder, f'overlay_{i+1}.png')
        overlay_trajectory_on_image(image_path, action, output_path, extend_factor)

def process_folder_actions(folder_path, output_folder, extend_factor):
    """Process all action .pkl files in a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    index = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            filepath = os.path.join(folder_path, filename)
            actions = load_pkl_file(filepath)
            image_size = (1080, 1920)  # Default image size, adjust if necessary
            for i, action in enumerate(actions):
                plot_trajectory(action, i, output_folder, image_size, extend_factor, index)
        index += 1

# Usage
file_path = '/home/ellina/Working/NFD/sweeping-piles-test/action/000000-1.pkl'
reference_image_folder = '/home/ellina/Working/NFD/output_images_train'  # Folder containing reference images to match sizes
output_folder = '/home/ellina/Working/NFD/output_actions'
output_folder_path = '/home/ellina/Working/NFD/output_actions_overlay'
action_folder = '/home/ellina/Working/NFD/sweeping-piles-test/action'

extend_factor = 0.3
# process_actions(file_path, output_folder, reference_image_folder, extend_factor)
process_actions_with_images(file_path, reference_image_folder, output_folder_path, extend_factor)
process_folder_actions(action_folder, output_folder, extend_factor)

