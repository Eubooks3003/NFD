# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data collection script."""

import os

from absl import app
from absl import flags

import numpy as np

from ravens import tasks
from ravens.dataset import Dataset
from ravens.environments.environment import ContinuousEnvironment
from ravens.environments.environment import Environment

flags.DEFINE_string('assets_root', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'towers-of-hanoi', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')
flags.DEFINE_bool('continuous', False, '')
flags.DEFINE_integer('steps_per_seg', 3, '')

FLAGS = flags.FLAGS

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import os
from torchvision.utils import save_image
import cv2
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import distance_transform_edt
from torch import Tensor

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 4)
        self.e2 = encoder_block(4, 8)
        self.e3 = encoder_block(8, 16)

        self.b = conv_block(16, 32)

        self.d1 = decoder_block(32, 16)
        self.d2 = decoder_block(16, 8)
        self.d3 = decoder_block(8, 4)

        self.outputs = nn.Conv2d(4, 1, kernel_size=1, padding=0)
    def forward(self, inputs):

        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b = self.b(p3)

        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        outputs = self.outputs(d3)
        return outputs

def frobenius_norm(y_pred, y_true):
    # Reshape tensors to [batch_size, C*H*W]
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    norm = torch.norm(y_pred_flat - y_true_flat, p='fro', dim=1)
    squared_frobenius_norm = norm ** 2
    # return squared_frobenius_norm.mean()

    return norm.mean()
def add_centered_rotated_rectangle_tensor(canvas, center, width, height, angle):
    """
    Add a centered and rotated rectangle to a PyTorch tensor canvas using a soft differentiable mask.

    Args:
    canvas: The PyTorch tensor representing the canvas (1, H, W).
    center: Tensor (x, y) representing the center of the rectangle.
    width: The width of the rectangle.
    height: The height of the rectangle.
    angle: The rotation angle of the rectangle in radians.
    """
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(
        torch.arange(canvas.shape[1], device=canvas.device, dtype=torch.float32), 
        torch.arange(canvas.shape[2], device=canvas.device, dtype=torch.float32), 
        indexing='ij'
    )

    # Adjust by center
    yy = yy - center[1]
    xx = xx - center[0]

    # Rotate the grid by the given angle
    x_rot = xx * torch.cos(angle) + yy * torch.sin(angle)
    y_rot = -xx * torch.sin(angle) + yy * torch.cos(angle)

    # Soft mask using sigmoid for smooth gradients
    mask_x = torch.sigmoid(-10 * (x_rot.abs() - width / 2))
    mask_y = torch.sigmoid(-10 * (y_rot.abs() - height / 2))
    mask = mask_x * mask_y

    # Directly modify the canvas using the smooth mask
    canvas = canvas + mask.unsqueeze(0)

    return canvas


def quaternion_to_rotation_matrix_torch(q, device):
    """
    Convert a quaternion to a rotation matrix using PyTorch.

    Args:
    q: Tensor (qx, qy, qz, qw) representing the quaternion.

    Returns:
    A 3x3 rotation matrix as a PyTorch tensor.
    """
    qx, qy, qz, qw = q
    return torch.tensor([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ], device=device)


def global_to_image_torch(global_coords, device):
    """
    Convert global coordinates to image coordinates using a differentiable method.

    Args:
    global_coords: Tensor representing global coordinates.

    Returns:
    A tensor representing image coordinates [x_pixel, y_pixel].
    """
    # Ensure global_coords requires_grad=True if intended to be differentiable
    data_position = torch.tensor([0.5, 0, 0.3], device=device)
    data_rotation = torch.tensor([0, torch.pi, -torch.pi / 2], device=device)
    image_size = (369, 492)
    intrinsics = torch.tensor([450., 0, 320., 0, 450., 240., 0, 0, 1], device=device)

    # Convert Euler angles to quaternion
    quaternion = torch.tensor(p.getQuaternionFromEuler(data_rotation.tolist()), device=device)
    rotation_matrix = quaternion_to_rotation_matrix_torch(quaternion, device)

    # Create transformation matrix
    transformation_matrix = torch.eye(4, device=device)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = data_position

    fx, fy, cx, cy = 450, 450, 320, 240
    scale_x = image_size[1] / 640.0
    scale_y = image_size[0] / 480.0

    K = torch.tensor([
        [fx * scale_x, 0, cx * scale_x],
        [0, fy * scale_y, cy * scale_y],
        [0, 0, 1]
    ], dtype=torch.float, device=device)

    # Instead of creating a new tensor, modify global_coords directly to adjust it
    adjusted_global_coords = torch.stack([
        global_coords[1],           # Swap Y to X
        global_coords[0] - 0.5,     # Swap X to Y and adjust
        global_coords[2]            # Z remains the same
    ])
    
    # Convert adjusted global coordinates to homogeneous coordinates
    global_coords_homogeneous = torch.cat([adjusted_global_coords, torch.tensor([1.0], dtype=torch.float, device=device)])

    camera_coords = global_coords_homogeneous[:2]
    camera_coords = torch.cat([camera_coords, torch.tensor([1.0], dtype=torch.float, device=device)])
    image_coords_homogeneous = torch.matmul(K, camera_coords)
    image_coords = image_coords_homogeneous[:2] / image_coords_homogeneous[2]

    # Ensure the coordinates are within image bounds
    x_pixel = torch.clamp(image_coords[0], 0, image_size[1] - 1)
    y_pixel = torch.clamp(image_coords[1], 0, image_size[0] - 1)
    
    return torch.stack([x_pixel, y_pixel])

import torch
import torch.nn.functional as F

def plot_trajectory(pose0_position: torch.Tensor, pose1_position: torch.Tensor, rect_height: int, rect_width: int, image_size=(369, 492), output_size=(256, 256)):
    """Plot extended 2D trajectory with individual rectangle images in a differentiable way."""

    # Assuming `global_to_image_torch` is a differentiable function that outputs PyTorch tensors
    pose0_image_coords = global_to_image_torch(pose0_position, device=pose0_position.device)
    pose1_image_coords = global_to_image_torch(pose1_position, device=pose1_position.device)

    # Using PyTorch operations
    direction = pose1_image_coords - pose0_image_coords
    norm_direction = direction / torch.norm(direction)

    positions = torch.stack([pose0_image_coords, pose1_image_coords], dim=0)
    angle = torch.atan2(direction[1], direction[0]) + torch.pi / 2


    # Create blank canvases
    canvases = []

    for pos in positions:
        # Create a blank canvas with the specified image size
        canvas = torch.zeros((1, image_size[0], image_size[1]), device=pose0_position.device, requires_grad=True)  # Single channel for grayscale image

        # Add the rectangle to the canvas using a differentiable function
        canvas = add_centered_rotated_rectangle_tensor(canvas, pos, rect_width, rect_height, angle)
        # Resize the canvas using differentiable interpolation
        canvas_resized = F.interpolate(canvas.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False).squeeze(0)

        # Normalize the resized canvas
        canvas_resized = (canvas_resized - 0.5) / 0.5
        canvases.append(canvas_resized)

    return canvases



def fill_goal_region(mask_green):
    """
    Fill the goal region in the green mask where only the border is highlighted.
    :param mask_green: Input binary mask where the goal's border is highlighted.
    :return: Binary mask with the entire goal region filled in.
    """
    # Create a copy of the mask to use for flood filling
    filled_mask = mask_green.copy()

    # Find contours to locate an internal point for flood filling
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming there is one main contour (the goal)
    if contours:
        # Get the moments to calculate the centroid of the contour
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Fallback if the moments calculation fails
            cX, cY = contours[0][0][0]

        # Flood fill from the centroid
        cv2.floodFill(filled_mask, None, (cX, cY), 255)

    # Combine the filled region with the original mask (in case there are other parts of the mask)
    final_mask = cv2.bitwise_or(filled_mask, mask_green)

    return final_mask

def visualize_hsv_channels(hsv_image, output_folder='/home/ellina/Working/NFD/overlay_images', file_name='color_tensor.png'):
    """
    Visualize the separate HSV channels to help tune the red color thresholds.
    :param hsv_image: Input HSV image.
    """
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(h_channel, cmap='hsv')
    plt.title('Hue Channel')

    plt.subplot(1, 3, 2)
    plt.imshow(s_channel, cmap='gray')
    plt.title('Saturation Channel')

    plt.subplot(1, 3, 3)
    plt.imshow(v_channel, cmap='gray')
    plt.title('Value Channel')

    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory
    print(f"HSV channel visualization saved to {output_path}")

def numpy_array_to_bgr_image(np_array):
    """
    Convert a NumPy array in RGB format to a BGR format, and ensure it is in the correct dtype.
    Expects the array to be in the range [0, 1] or [0, 255]. Converts it to uint8.
    
    :param np_array: Input NumPy array in (H, W, C) format, possibly in RGB.
    :return: NumPy array in (H, W, C) format in BGR color space, with dtype uint8.
    """
    # Check if the image is in float format and needs conversion to uint8
    if np_array.max() <= 1.0:
        np_array = (np_array * 255).astype(np.uint8)  # Scale to [0, 255]
    else:
        np_array = np_array.astype(np.uint8)

    # Convert from RGB (default NumPy array format) to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    return image_bgr

def segment_color_objects_from_numpy(np_array):

    # Convert the NumPy array to a format compatible with OpenCV
    image_bgr = numpy_array_to_bgr_image(np_array)

    # Proceed with the same segmentation code
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

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

    return mask_red, mask_green

def check_red_segmentation(mask_red):
    """
    Check if the red mask segments anything by counting the number of non-zero pixels.
    :param mask_red: Red mask as a NumPy array.
    :return: Boolean indicating if red objects are detected.
    """
    red_pixel_count = np.count_nonzero(mask_red)
    if red_pixel_count > 0:
        print(f"Red segmentation detected {red_pixel_count} red pixels.")
        return True
    else:
        print("No red objects detected.")
        return False

def transform_color(color, image_size):
    """
    Transform the color array into a PyTorch tensor with the necessary preprocessing.
    :param color: NumPy array representing the color image.
    :param color_image_size: The expected size of the color image (height, width, channels).
    :return: Transformed color image as a PyTorch tensor.
    """
    # Ensure the color array is the right type and shape
    color_image_size = (image_size[0], image_size[1], 3)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # Remove alpha channel if it exists (retain only RGB)

    # print("Color: ", color)
    # print("Color Shape; ", color.shape)

    output_folder = '/home/ellina/Working/NFD/overlay_images'

    visualize_and_save_tensor(color, output_folder, file_name='color_tensor.png')

    mask_red, mask_green = segment_color_objects_from_numpy(color)

    visualize_and_save_tensor(mask_red, output_folder, file_name='red_mask.png')
    visualize_and_save_tensor(mask_green, output_folder, file_name='green_mask.png')
    check_red_segmentation(mask_red)

    print("Red Mask Shape: ", mask_red.shape)
    mask_red_pil = Image.fromarray(mask_red)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images
        transforms.ToTensor(),          # Convert images to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for 3 stacked grayscale images
    ])

    red_mask_tensor = transform(mask_red_pil)

    # print("red Mask Tensor: ", red_mask_tensor)

    return red_mask_tensor
def fill_polygon_on_canvas(canvas, image_corners):
    """
    Fills a polygon defined by the corners on the canvas using PyTorch operations.

    Args:
    - canvas: The PyTorch tensor representing the canvas (1, H, W).
    - image_corners: Tensor containing the image coordinates of the polygon corners.

    Returns:
    Updated canvas tensor with the polygon filled.
    """
    # Get image size from the canvas
    height, width = canvas.shape[1], canvas.shape[2]

    # Create a grid of coordinates for the height and width of the canvas
    yy, xx = torch.meshgrid(
        torch.arange(height, device=canvas.device),
        torch.arange(width, device=canvas.device),
        indexing='ij'
    )

    yy = yy.float()
    xx = xx.float()
    mask = torch.ones((height, width), dtype=torch.bool, device=canvas.device)

    # Compute ray-casting: count number of edge crossings for each point
    num_corners = len(image_corners)
    for i in range(num_corners):
        p1 = image_corners[i]
        p2 = image_corners[(i + 1) % num_corners]

        # Check if the ray intersects the edge (p1, p2)
        condition = (((yy > p1[1]) != (yy > p2[1])) & 
                     (xx < (p2[0] - p1[0]) * (yy - p1[1]) / (p2[1] - p1[1] + 1e-6) + p1[0]))

        # Toggle mask where the ray intersects an odd number of times
        mask ^= condition  # XOR to toggle between inside and outside, both are boolean

    # Convert mask to float for applying to canvas
    mask = mask.float()

    # Set the mask back to the canvas (0 inside the polygon, 1 outside)
    new_canvas = torch.where(mask == 0, torch.tensor(0.0, device=canvas.device), canvas)

    return canvas


def create_goal_mask_edf(goal, device, image_size=(369, 492), output_size=(256, 256)):
    """
    Create a goal mask and its Euclidean Distance Field (EDF) in a differentiable way using PyTorch.

    Args:
    - goal: Tuple containing the goal information (center_position, rotation), square_size.
    - device: PyTorch device (CPU or CUDA) for the calculations.
    - image_size: Tuple representing the size of the image.
    - output_size: Tuple representing the size of the output image.

    Returns:
    - mask_transformed: Differentiable mask tensor.
    - edf_transformed: Differentiable EDF tensor.
    """

    # Unpack the goal information
    (center_position, rotation), square_size = goal
    square_center = torch.tensor(center_position, dtype=torch.float, device=device, requires_grad=True)  # Ensure requires_grad=True
    square_width, square_height = square_size[:2]  # width, height

    # Calculate the unrotated corners of the square in global coordinates using PyTorch tensors
    half_width = square_width / 2
    half_height = square_height / 2

    corners = torch.tensor([
        [square_center[0] - half_width, square_center[1] - half_height, 0],
        [square_center[0] + half_width, square_center[1] - half_height, 0],
        [square_center[0] + half_width, square_center[1] + half_height, 0],
        [square_center[0] - half_width, square_center[1] + half_height, 0]
    ], dtype=torch.float32, device=device, requires_grad=True)  # Ensure requires_grad=True

    # Apply the rotation using the quaternion
    rotation_matrix = quaternion_to_rotation_matrix_torch(rotation, device)
    
    # Correct use of `.to()` and `.requires_grad_()`
    rotation_matrix = rotation_matrix.to(dtype=torch.float32)
    rotation_matrix.requires_grad_()  # Set requires_grad=True after converting dtype

    # Ensure all tensors are Float for the matrix multiplication
    corners = corners.to(dtype=torch.float32)
    square_center = square_center.to(dtype=torch.float32)
    rotated_corners = torch.matmul(rotation_matrix, (corners - square_center).T).T + square_center

    # Convert the rotated corners to image coordinates using PyTorch operations
    image_corners = torch.stack([global_to_image_torch(corner, device) for corner in rotated_corners])

    # Create a blank mask canvas
    mask = torch.ones((1, image_size[0], image_size[1]), dtype=torch.float, device=device, requires_grad=True)

    mask = fill_polygon_on_canvas(mask, image_corners)
    inverted_mask = 1 - mask

    if (inverted_mask == inverted_mask[0, 0]).all():
        # print("inverted_mask contains identical points, leading to potential issues in torch.cdist.")
        inverted_mask += torch.eye(inverted_mask.size(-2), inverted_mask.size(-1), device=device)

    # Approximate Euclidean Distance Field using PyTorch
    try:
        edf = torch.cdist(inverted_mask.unsqueeze(0).unsqueeze(0), inverted_mask.unsqueeze(0).unsqueeze(0)).squeeze()
    except RuntimeError as e:
        print("Error during torch.cdist computation:", e)
        return mask, None

    # Safeguard normalization to avoid division by zero
    if edf.max() == edf.min():
        print("Warning: edf max equals edf min, normalization will lead to NaN values.")
        edf_normalized = torch.zeros_like(edf)  # or any default value that makes sense
    else:
        edf_normalized = (edf - edf.min()) / (edf.max() - edf.min())
    
    edf_normalized = edf_normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Ensure mask also has 4 dimensions before interpolation
    mask = mask.unsqueeze(0)  # Add batch dimension to match interpolation format

    # Resize both mask and EDF to the output size using differentiable interpolation
    mask_resized = F.interpolate(mask, size=output_size, mode='bilinear', align_corners=False).squeeze(0)
    edf_resized = F.interpolate(edf_normalized, size=output_size, mode='bilinear', align_corners=False).squeeze(0)

    # Normalize and transform into tensors
    mask_transformed = (mask_resized - 0.5) / 0.5
    edf_transformed = (edf_resized - 0.5) / 0.5

    return mask_transformed, edf_transformed

def euclidean_distance_field(mask):
    """
    Compute the Euclidean Distance Field (EDF) of the target zone mask.
    :param mask: Binary mask (H x W) with 1 inside the target zone and 0 outside.
    :return: Euclidean Distance Field of the mask.
    """
    # Use PyTorch's distance transform functionality if available, or implement manually
    return F.distance_transform_edt(mask.float())

def l_goal(sT, goal_mask, goal_edf_mask, alpha1=1.0, alpha2=2.0):
    # Check requires_grad of inputs


    loss = alpha1 * torch.sum(goal_edf_mask * sT) - alpha2 * torch.sum(goal_mask * sT)

    return loss

import os
import torch
from torchvision.utils import save_image
import numpy as np

def overlay_colored_images_and_save(combined_image, goal_mask, output_folder, file_name='overlay.png', alpha=0.5):
    """
    Saves the overlay of the combined_image with the goal_mask, each in a different color using torchvision's save_image.

    Args:
        combined_image (torch.Tensor): Tensor of shape (3, H, W) representing 3 grayscale images.
        goal_mask (torch.Tensor): Tensor of shape (H, W) representing the goal mask.
        output_folder (str): Folder to save the images.
        file_name (str): Name of the file to save the image as.
        alpha (float): Transparency level for the overlay. 0 means fully transparent, 1 means fully opaque.
    """
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Ensure combined_image and goal_mask are on CPU
    combined_image = combined_image.detach().cpu()
    goal_mask = goal_mask.detach().cpu()

    # Extract the individual grayscale images from combined_image
    img1 = combined_image[0]  # First grayscale image
    img2 = combined_image[1]  # Second grayscale image
    img3 = combined_image[2]  # Third grayscale image

    # Normalize each image for visualization
    def normalize_img(img):
        return (img - img.min()) / (img.max() - img.min())

    img1 = normalize_img(img1)
    img2 = normalize_img(img2)
    img3 = normalize_img(img3)

    # Create a 3-channel colored image where each channel represents one of the combined images
    colored_image = torch.stack([img1, img2, img3], dim=0)

    # Create a mask overlay in yellow (red + green)
    goal_mask_colored = torch.stack([goal_mask, goal_mask, torch.zeros_like(goal_mask)], dim=0)  # Red + Green = Yellow

    # Blend the goal mask into the colored image with the specified alpha
    final_image = (1 - alpha) * colored_image + alpha * goal_mask_colored

    # Save the final blended image
    output_path = os.path.join(output_folder, file_name)
    save_image(final_image, output_path)

    print(f"Overlay saved to {output_path}")

import os
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
def visualize_and_save_tensor(color_tensor, output_folder='/home/ellina/Working/NFD/overlay_images', file_name='color_tensor.png'):
    """
    Visualizes and saves the single-channel color_tensor.
    
    Args:
        color_tensor (torch.Tensor or np.ndarray): Tensor of shape (1, H, W) or NumPy array of shape (H, W).
        output_folder (str): Folder to save the images.
        file_name (str): Name of the file to save the image as.
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Case 1: If color_tensor is a PyTorch tensor
    if isinstance(color_tensor, torch.Tensor):
        # If it's a PyTorch tensor, ensure it's detached and on the CPU
        color_tensor = color_tensor.detach().cpu()

        # Check if it's a single-channel tensor of shape (1, H, W)
        if color_tensor.dim() == 3 and color_tensor.shape[0] == 1:
            output_path = os.path.join(output_folder, file_name)
            # Save using torchvision's save_image for PyTorch tensors
            save_image(color_tensor, output_path)
            print(f"Color tensor saved to {output_path}")

        # Convert to NumPy for visualization
        color_tensor = color_tensor.squeeze(0).numpy()  # Convert (1, H, W) -> (H, W)

    # Case 2: If color_tensor is a NumPy array
    elif isinstance(color_tensor, np.ndarray):
        # If it's a NumPy array, ensure it's of shape (H, W) for visualization
        if color_tensor.ndim == 3 and color_tensor.shape[0] == 1:
            color_tensor = np.squeeze(color_tensor, axis=0)  # Convert (1, H, W) -> (H, W)

        # Optionally save using PIL if it's a NumPy array
        output_path = os.path.join(output_folder, file_name)
        plt.imsave(output_path, color_tensor, cmap='gray')
        print(f"Color array saved to {output_path}")

    # Visualize the color_tensor (whether it's a NumPy array or a Tensor converted to NumPy)
    plt.figure(figsize=(6, 6))
    plt.imshow(color_tensor, cmap='gray')
    plt.title('Color Tensor Visualization')
    plt.axis('off')


def optimize_trajectory_no_obstacles(model, x0, G, T, learning_rate=1e-2, max_iters=1000, num_iteration = 0, device = None):
    """
    Optimize the trajectory to minimize the objective function with no obstacles.
    
    :param model: Learned dynamics model function f_theta(xt, [r(xt), r(xt+1)]) -> st+1.
    :param x0: Initial state (image or state vector).
    :param G: Target region mask (H x W).
    :param T: Horizon length.
    :param learning_rate: Learning rate for gradient descent.
    :param max_iters: Maximum number of optimization iterations.
    :return: Optimized trajectory.
    """
    # Initialize trajectory
    xt = x0
    output_folder = '/home/ellina/Working/NFD/overlay_images'

    image_size = (480, 640)
    color_tensor = transform_color(xt, image_size)
    visualize_and_save_tensor(color_tensor, output_folder, file_name=f'color_tensor_{num_iteration}.png')

    pose0_series = [torch.tensor([0.4, 0.0, 0.0], requires_grad=True) for _ in range(T)]
    pose1_series = [torch.tensor([0.5, 0.0, 0.0], requires_grad=True) for _ in range(T)]

    optimizer = optim.Adam(pose0_series + pose1_series, lr = learning_rate)

    rect_height = 50
    rect_width = 100


    final_combined_image = None
    for iter in range(40):
        optimizer.zero_grad()
        action_tensors = plot_trajectory(pose0_series[0], pose1_series[0], rect_height, rect_width)
        combined_image = torch.cat([color_tensor, action_tensors[0], action_tensors[1]], dim=0)
        goal_mask, edf_mask = create_goal_mask_edf(G, device)
        # Initialize the total loss
        total_loss = torch.tensor(0.0, device=device)

        # Forward pass through time
        device = next(model.parameters()).device 
        for t in range(T):
            if t == 0:
              combined_image = combined_image.to(device) 
              combined_image = combined_image.unsqueeze(0)
              st_next = model(combined_image)
              combined_image = st_next
              goal_mask = goal_mask.to(device)
              edf_mask = edf_mask.to(device)

              total_loss += l_goal(combined_image, goal_mask, edf_mask)
            else:
              action_tensors = plot_trajectory(pose0_series[t], pose1_series[t], rect_height, rect_width)
            
              action_tensors[0] = action_tensors[0].to(device)
              action_tensors[1] = action_tensors[1].to(device)

              combined_image = combined_image.squeeze(0)
              combined_image = torch.cat([combined_image, action_tensors[0], action_tensors[1]], dim=0)
              combined_image = combined_image.unsqueeze(0)

              st_next = model(combined_image)
              combined_image = st_next  # Update the state for the next time step

              total_loss += l_goal(combined_image, goal_mask, edf_mask)

        # Perform the backward pass

        total_loss.backward()

        # Update the trajectory with the optimizer
        optimizer.step()


        # Optional: Print the loss at each iteration
        if iter % 10 == 0:
            print(f"Iteration {iter}, Total Loss: {total_loss.item()}")

    action_tensors = plot_trajectory(pose0_series[0], pose1_series[0], rect_height, rect_width)
    combined_image = torch.cat([color_tensor, action_tensors[0], action_tensors[1]], dim=0)

    print("Combined Image Shape: ", combined_image.shape)

    overlay_colored_images_and_save(combined_image, goal_mask, output_folder, file_name=f'overlay_{num_iteration}.png', alpha=0.5)
    return [pose0_series[0], pose1_series[0]]

def debug():
    # Example simple setup for debugging
    pose0_position = torch.tensor([0.4, 0.0, 0.0], requires_grad=True)
    pose1_position = torch.tensor([0.5, 0.0, 0.0], requires_grad=True)
    rect_height = 50
    rect_width = 100

    # Example simplified trajectory
    action_tensors = plot_trajectory(pose0_position, pose1_position, rect_height, rect_width)
    # Simplified loss for debugging
    loss = action_tensors[0].sum() + action_tensors[1].sum()  # Simple sum to ensure gradient flow
    loss.backward()

    print(f"pose0_position grad: {pose0_position.grad}")
    print(f"pose1_position grad: {pose1_position.grad}")

def test_gradient_flow():
    # Define test inputs
    canvas = torch.zeros((1, 369, 492), requires_grad=True)
    center = torch.tensor([184.5, 246.0], requires_grad=True)  # Assume some center in the middle
    width = torch.tensor(100.0, requires_grad=True)
    height = torch.tensor(50.0, requires_grad=True)
    angle = torch.tensor(torch.pi / 4, requires_grad=True)  # 45 degrees
    output_canvas = add_centered_rotated_rectangle_tensor(canvas, center, width, height, angle)

    print(f"Output canvas requires_grad: {output_canvas.requires_grad}")
    print(f"Output canvas grad_fn: {output_canvas.grad_fn}")

    # Use output in a downstream function
    # Example: simply sum all elements to create a loss
    downstream_output = output_canvas.mean()  # Example downstream operation
    loss = downstream_output.sum()  # Example loss computation

    # Perform backpropagation
    loss.backward()

    # Check if gradients are non-zero
    print(f"Gradient of center: {center.grad}")
    print(f"Gradient of width: {width.grad}")
    print(f"Gradient of height: {height.grad}")
    print(f"Gradient of angle: {angle.grad}")
    print(f"Gradient of canvas: {canvas.grad}")

def debug_create_goal_mask_edf():
    # Define test inputs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    goal = (((0.478125, 0.04062500000000002, 0.0), (0.0, -0.0, 0.7800583048932599, -0.6257068330832372)), (0.12, 0.12, 0))

    # Call the function
    mask_transformed, edf_transformed = create_goal_mask_edf(goal, device)

    # Use output in a downstream function
    # Example: simply sum all elements to create a loss
    loss = mask_transformed.sum() + edf_transformed.sum()  # Example loss computation

    # Perform backpropagation
    loss.backward()

    # Check if gradients are non-zero for the inputs
    # Since we do not have direct access to corners and rotation_matrix as they are intermediate, we test main inputs
    square_center = torch.tensor(goal[0][0], dtype=torch.float, device=device, requires_grad=True)
    square_width, square_height = goal[1][:2]


class Agent:
    def __init__(self, model, T, learning_rate=1e-2, max_iters=1000, device = None):
        self.model = model
        self.T = T  # Horizon length
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.optimized_trajectory = None
        self.step_index = 0
        self.device = device
    
    def optimize_trajectory(self, initial_state, G):
        # Optimize the trajectory once at the start of each episode.
        self.optimized_trajectory = optimize_trajectory_no_obstacles(
            model=self.model,
            x0=initial_state,
            G=G,
            T=self.T,
            learning_rate=self.learning_rate,
            max_iters=self.max_iters,
            num_iteration = self.step_index,
            device = self.device
        )
        self.step_index += 1

    def act(self, obs, info):
        if self.optimized_trajectory is None:
            # If the trajectory hasn't been optimized yet or we're out of steps, return a default action.
            return np.zeros((2,))  # Replace with a default action for your environment

        # Extract the action from the optimized trajectory
        [pose0, pose1] = self.optimized_trajectory
        pose0_np = pose0.detach().cpu().numpy()
        pose1_np = pose1.detach().cpu().numpy()
        action = {
            'pose0': (pose0_np, np.array([0, 0, 0, 1])),  # Pose0 position and orientation
            'pose1': (pose1_np, np.array([0, 0, 0, 1]))   # Pose1 position and orientation
        }
        return action


def main(unused_argv):

  # Initialize environment and task.
  env_cls = ContinuousEnvironment if FLAGS.continuous else Environment
  env = env_cls(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
  task = tasks.names[FLAGS.task](continuous=FLAGS.continuous)

  task.mode = FLAGS.mode

  # Initialize scripted oracle agent and dataset.
  print("Continuous: ", FLAGS.continuous)
  agent = task.oracle(env, steps_per_seg=FLAGS.steps_per_seg)

  checkpoint_path = '/home/ellina/Working/NFD/micro-actions/model_checkpoint_best.pth'

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Unet().to(device)
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))

  max_steps = task.max_steps
  
  T = 10
  learning_rate = 1e-2
  agent = Agent(model=model, T=T, learning_rate=learning_rate, max_iters=max_steps, device = device)
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2

  # Determine max steps per episode.
  max_steps = task.max_steps
  if FLAGS.continuous:
    max_steps *= (FLAGS.steps_per_seg * agent.num_poses)

  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < FLAGS.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    obs = env.reset()

    print("Goal length: ", len(task.goals))
    print("Goal: ", task.goals[0][6][1][0])

    Goal = task.goals[0][6][1][0]

    obs_color_array = np.array(obs['color'])
    print("Color Dim: ", obs_color_array.shape)
    # color_tensor = torch.tensor(obs['color'], dtype=torch.float32, device=device).permute(2, 0, 1)  # (C, H, W) format
    agent.optimize_trajectory(obs['color'], Goal) 

    info = None
    reward = 0
    for _ in range(max_steps):
      act = agent.act(obs, info)
      print("Action: ", act)
      episode.append((obs, act, reward, info))
      obs, reward, done, info = env.step(act)
      agent.optimize_trajectory(obs['color'], Goal) 
      total_reward += reward
      print(f'Total Reward: {total_reward} Done: {done}')
      if done:
        break
    episode.append((obs, None, reward, info))

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    # if total_reward > 0.99:
    dataset.add(seed, episode)

if __name__ == '__main__':
#   test_gradient_flow()
#   debug()
#   debug_create_goal_mask_edf()
  app.run(main)
