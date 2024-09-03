import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2  # Import OpenCV for handling image operations
import re
import pybullet as p

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

                return image.shape # Returns (height, width)
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
def plot_trajectory(action, image_index, output_folder, image_size, extend_factor, index, rect_height, rect_width):
    """Plot extended 2D trajectory with individual rectangle images."""
    if action is None:
        print("No action provided for plot.")
        return

    fig, ax = plt.subplots(figsize=(image_size[1] / 100, image_size[0] / 100))
    ax.set_xlim([0, image_size[1]])
    ax.set_ylim([image_size[0], 0]) 
    print("Action: ", action)
    pose0_position = action['pose0'][0]
    pose1_position = action['pose1'][0]

    print("Initial start: ", pose0_position)
    print("Intial End: ", pose1_position)

    transform_x = lambda y: int(y * image_size[1] + image_size[1] / 2)
    transform_y = lambda x: int((x - 0.5) * image_size[0] + image_size[0] / 2)

    # transform_x = lambda x: x * image_size[1]
    # transform_y = lambda y: y * image_size[0]

    pose0_image_coords = global_to_image(pose0_position)
    pose1_image_coords = global_to_image(pose1_position)
    original_xs = [pose0_image_coords[0], pose1_image_coords[0]]
    original_ys = [pose0_image_coords[1],pose1_image_coords[1]]

    direction = np.array([original_xs[1] - original_xs[0], original_ys[1] - original_ys[0]])
    norm_direction = direction / np.linalg.norm(direction)
    extended_start = np.array([original_xs[0], original_ys[0]]) - norm_direction * extend_factor * np.linalg.norm(direction)
    extended_end = np.array([original_xs[1], original_ys[1]]) + norm_direction * extend_factor * np.linalg.norm(direction)

    # ax.plot([extended_start[0], extended_end[0]], [extended_start[1], extended_end[1]], 'go-', label= f'Camera_1_Index_{index + 1}_Image_{image_index + 1}.png')
    positions = [extended_start, extended_end]
    angles = [np.degrees(np.arctan2(direction[1], direction[0]) + np.pi / 2) for _ in positions]

    print("Extended Start Plot Trajectory: ", extended_start)
    print("Extended End Plot Trajectory: ", extended_end)

    # Create individual plots for each rectangle
    for i, pos in enumerate(positions):
        fig, ax = plt.subplots(figsize=(image_size[1] / 100, image_size[0] / 100))
        fig.patch.set_facecolor('black') 

        # Set the limits of x and y to match the size of your image and invert the y-axis
        ax.set_xlim([0, image_size[1]])
        ax.set_ylim([image_size[0], 0])  # Maintain this inversion
        
        # Plot the line and add the rectangle
        # ax.plot([extended_start[0], extended_end[0]], [extended_start[1], extended_end[1]], 'go-')
        add_centered_rotated_rectangle(ax, (pos[0], pos[1]), rect_width, rect_height, angles[i])

        ax.axis('off')  # Turn off the axis
        
        # Save the figure
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

# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])


def global_to_image(global_coords):
    data_position = (0.5, 0, 0.3)
    data_rotation = (0, np.pi, -np.pi/2)
    image_size = (369, 492)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Convert Euler angles to quaternion
    quaternion = p.getQuaternionFromEuler(data_rotation)
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)

    # Create transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = data_position

    fx, fy, cx, cy = 450, 450, 320, 240
    scale_x = image_size[1] / 640.0
    scale_y = image_size[0] / 480.0

    # Define intrinsic matrix
    K = np.array([
        [fx * scale_x, 0, cx * scale_x],
        [0, fy * scale_y, cy * scale_y],
        [0, 0, 1]
    ])
    adjusted_global_coords = np.array([
        global_coords[1],          # Swap Y to X
        global_coords[0] - 0.5,    # Swap X to Y and adjust
        global_coords[2]           # Z remains the same
    ])
    
    # Convert adjusted global coordinates to homogeneous coordinates
    global_coords_homogeneous = np.append(adjusted_global_coords, 1)

    print("Global Coords Homogenous: ", global_coords_homogeneous)
    
    # Transform to camera coordinates
    # camera_coords_homogeneous = np.dot(transformation_matrix, global_coords_homogeneous)

    # print("Camera Coords Homogenous: ", camera_coords_homogeneous)
    # camera_coords = camera_coords_homogeneous[:2]
    
    # Project to image plane
    # print("Camera Coordinates: ", camera_coords)
    camera_coords = global_coords_homogeneous[:2]
    camera_coords = np.append(camera_coords, 1)
    image_coords_homogeneous = np.dot(K, camera_coords)
    # print("Image Coords Homogenous: ", image_coords_homogeneous)
    image_coords = image_coords_homogeneous[:2] / image_coords_homogeneous[2]
    
    # Ensure the coordinates are within image bounds
    x_pixel = int(np.clip(image_coords[0], 0, image_size[1] - 1))
    y_pixel = int(np.clip(image_coords[1], 0, image_size[0] - 1))
    
    return (x_pixel, y_pixel)

def overlay_trajectory_on_image(image_path, action, output_path, extend_factor, image_size, rect_height, rect_width):
    """Overlay extended trajectory on an image, draw rectangles at the new endpoints, and save the result."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image:", image_path)
        return

    if action is None:
        print("No action provided for overlay.")
        return
    print("Image Size: ", image_size)
    # Define the color for the trajectory and rectangles (BGR format)
    line_color = (0, 255, 0)  # Green for line
    rect_color_init = (255, 0, 0)  # Blue
    rect_color_final = (0, 0, 255) # Red
    point_color = (255, 255, 0)  # Cyan for the point (0.5, 0)
    random_point_color = (0, 255, 255)

    rect_color = [rect_color_init, rect_color_final]
    thickness = 2  # Line and rectangle thickness

    # Transform coordinates to match image dimensions centered at (0.5, 0)
    pose0_position = action['pose0'][0]
    pose1_position = action['pose1'][0]

    print("Initial start: ", pose0_position)
    print("Intial End: ", pose1_position)

    transform_x = lambda y: int(y * image_size[1] + image_size[1] / 2)
    transform_y = lambda x: int((x - 0.5) * image_size[0] + image_size[0] / 2)

    # transform_x = lambda x: x * image_size[1]
    # transform_y = lambda y: y * image_size[0]

    pose0_image_coords = global_to_image(pose0_position)
    pose1_image_coords = global_to_image(pose1_position)
    original_xs = [pose0_image_coords[0], pose1_image_coords[0]]
    original_ys = [pose0_image_coords[1],pose1_image_coords[1]]


    direction = np.array([original_xs[1] - original_xs[0], original_ys[1] - original_ys[0]])
    #norm_direction = direction / np.linalg.norm(direction)
    extended_start = np.array([original_xs[0], original_ys[0]]) # - norm_direction * extend_factor * np.linalg.norm(direction)
    extended_end = np.array([original_xs[1], original_ys[1]]) # + norm_direction * extend_factor * np.linalg.norm(direction)

    # ax.plot([extended_start[0], extended_end[0]], [extended_start[1], extended_end[1]], 'go-', label= f'Camera_1_Index_{index + 1}_Image_{image_index + 1}.png')
    positions = [extended_start, extended_end]
    angles = [np.degrees(np.arctan2(direction[1], direction[0]) + np.pi / 2) for _ in positions]


    extended_start = (int(extended_start[0]), int(extended_start[1]))
    extended_end = (int(extended_end[0]), int(extended_end[1]))

    print("Extended Start Overlay: ", extended_start)
    print("Extended End Overlay: ", extended_end)

    # Draw the extended trajectory line
    cv2.line(image, extended_start, extended_end, line_color, thickness)

    # Draw rectangles at the new endpoints

    for i in range(len([extended_start, extended_end])):
        pos = [extended_start, extended_end][i]
        top_left = (pos[0] - rect_width // 2, pos[1] - rect_height // 2)
        bottom_right = (pos[0] + rect_width // 2, pos[1] + rect_height // 2)
        cv2.rectangle(image, top_left, bottom_right, rect_color[i], thickness)
    
    center_point = (transform_x(0), transform_y(0.5))
    cv2.circle(image, center_point, 5, point_color, -1)  # Draw a solid circle

    random_point = (transform_x(0), transform_y(0.4))
    cv2.circle(image, random_point, 5, random_point_color, -1)
    print("Center point (0.5, 0) on image:", center_point)
    
    
    # Save the image with the trajectory and rectangles
    result = cv2.imwrite(output_path, image)
    if result:
        print(f"Successfully saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")


import shutil

def clear_folder(folder):
    # Check if the folder exists
    if not os.path.exists(folder):
        # Create the folder if it does not exist
        os.makedirs(folder)
    else:
        # If the folder exists, delete its contents
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

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

def process_folder_actions(folder_path, output_folder, extend_factor, overlay_folder, image_folder):
    """Process all action .pkl files in a folder."""
    
    clear_folder(output_folder)
    clear_folder(overlay_folder)

    image_paths = get_images(image_folder)
    image_size = get_image_size(image_folder)


    index = 0
    rect_height = 50
    rect_width = 100
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            filepath = os.path.join(folder_path, filename)
            actions = load_pkl_file(filepath)
            image_size = (369, 492) 
            for i, (action, image_path) in enumerate(zip(actions, image_paths)):
                plot_trajectory(action, i, output_folder, image_size, extend_factor, index, rect_height, rect_width)
                output_path = os.path.join(overlay_folder, f'overlay_{index}_{i+1}.png')
                overlay_trajectory_on_image(image_path, action, output_path, extend_factor, image_size, rect_height, rect_width)
        index += 1

# 5 Blocks
file_path = '/home/ellina/Working/NFD/sweeping-piles-test/action/000000-1.pkl'
reference_image_folder = '/home/ellina/Working/NFD/output_images_test'  # Folder containing reference images to match sizes
output_folder = '/home/ellina/Working/NFD/output_actions_test'
overlay_folder = '/home/ellina/Working/NFD/output_actions_overlay_test'
output_folder_path = '/home/ellina/Working/NFD/output_actions_overlay_test'
action_folder = '/home/ellina/Working/NFD/sweeping-piles-test/action'

# 50 Blocks

# file_path = '/home/ellina/Working/NFD/sweeping-piles-train/action/000000-1.pkl'
# reference_image_folder = '/home/ellina/Working/NFD/output_images_train'  # Folder containing reference images to match sizes
# output_folder = '/home/ellina/Working/NFD/output_actions_train'
# overlay_folder = '/home/ellina/Working/NFD/output_actions_overlay_train'
# output_folder_path = '/home/ellina/Working/NFD/output_actions_overlay_train'
# action_folder = '/home/ellina/Working/NFD/sweeping-piles-train/action'

extend_factor = 0
# process_actions(file_path, output_folder, reference_image_folder, extend_factor)
# process_actions_with_images(file_path, reference_image_folder, output_folder_path, extend_factor)
process_folder_actions(action_folder, output_folder, extend_factor, overlay_folder, reference_image_folder)


# global_coords = (0.5, 0, 0)  # Replace with your global coordinates
# image_coords = global_to_image(global_coords)
# print("Image Coordinates:", image_coords)