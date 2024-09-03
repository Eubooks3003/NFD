import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
import pybullet as p

import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import distance_transform_edt


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
    image_size = (480, 640)
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
    

    camera_coords = global_coords_homogeneous[:2]
    camera_coords = np.append(camera_coords, 1)
    image_coords_homogeneous = np.dot(K, camera_coords)
    # print("Image Coords Homogenous: ", image_coords_homogeneous)
    image_coords = image_coords_homogeneous[:2] / image_coords_homogeneous[2]
    
    # Ensure the coordinates are within image bounds
    x_pixel = int(np.clip(image_coords[0], 0, image_size[1] - 1))
    y_pixel = int(np.clip(image_coords[1], 0, image_size[0] - 1))
    
    return (x_pixel, y_pixel)

import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R

def create_goal_mask(goal, global_to_image, save_path_mask, save_path_edf, image_size):
    # Unpack the goal information

    image_size = (480, 640)

    (center_position, rotation) , square_size = goal
    square_center = center_position  # (x, y) position
    square_width, square_height = square_size[:2]  # width, height

    # Convert the center position to image coordinates
    image_center = np.array(global_to_image(square_center))

    # Calculate the unrotated corners of the square in global coordinates
    half_width = square_width / 2
    half_height = square_height / 2

    corners = np.array([
        [square_center[0] - half_width, square_center[1] - half_height, 0],
        [square_center[0] + half_width, square_center[1] - half_height, 0],
        [square_center[0] + half_width, square_center[1] + half_height, 0],
        [square_center[0] - half_width, square_center[1] + half_height, 0]
    ])

    # Apply the rotation using the quaternion
    r = R.from_quat(rotation)
    rotated_corners = r.apply(corners - square_center) + square_center

    # Convert the rotated corners to image coordinates
    image_corners = [global_to_image(corner) for corner in rotated_corners]

    mask = np.ones(image_size, dtype=np.uint8) * 255  # Initialize mask with 255 (1 in binary)

    # Draw the filled square in the mask (inside square = 0)
    cv2.fillPoly(mask, [np.array(image_corners, dtype=np.int32)], 0)

    # Save the mask as an image
    mask_image = Image.fromarray(mask)
    mask_image.save(save_path_mask)

    edf = distance_transform_edt(mask)

    # Normalize the EDF to [0, 255]
    edf_normalized = cv2.normalize(edf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save the EDF as an image
    edf_image = Image.fromarray(edf_normalized)
    edf_image.save(save_path_edf)

    # Apply the transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize images
        transforms.ToTensor(),          # Convert images to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single channel
    ])

    mask_transformed = transform(mask_image)
    edf_transformed = transform(edf_image)

    return mask_transformed, edf_transformed

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

    # Initialize the mask as zeros (filled inside the polygon)
    mask = torch.zeros((height, width), dtype=torch.bool, device=canvas.device)

    # Define the corners of the polygon (image_corners are already in image coordinates)
    for i in range(image_corners.shape[0]):
        p1 = image_corners[i]
        p2 = image_corners[(i + 1) % image_corners.shape[0]]

        # Fill the polygon using ray-casting algorithm for each edge
        mask |= (
            ((yy - p1[1]) * (p2[0] - p1[0]) > (xx - p1[0]) * (p2[1] - p1[1])) ^
            ((yy - p2[1]) * (p1[0] - p2[0]) > (xx - p2[0]) * (p1[1] - p2[1]))
        )

    # Apply the mask to the canvas to fill the polygon
    canvas[0, mask] = 1.0

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
    square_center = torch.tensor(center_position, dtype=torch.float, device=device)  # (x, y) position
    square_width, square_height = square_size[:2]  # width, height

    # Calculate the unrotated corners of the square in global coordinates using PyTorch tensors
    half_width = square_width / 2
    half_height = square_height / 2

    corners = torch.tensor([
        [square_center[0] - half_width, square_center[1] - half_height, 0],
        [square_center[0] + half_width, square_center[1] - half_height, 0],
        [square_center[0] + half_width, square_center[1] + half_height, 0],
        [square_center[0] - half_width, square_center[1] + half_height, 0]
    ], dtype=torch.float32, device=device)  # Convert to Float

    # Apply the rotation using the quaternion
    rotation_matrix = quaternion_to_rotation_matrix_torch(rotation, device)
    
    # Ensure all tensors are Float for the matrix multiplication
    corners = corners.to(dtype=torch.float32)
    square_center = square_center.to(dtype=torch.float32)
    rotation_matrix = rotation_matrix.to(dtype=torch.float32)
    rotated_corners = torch.matmul(rotation_matrix, (corners - square_center).T).T + square_center

    # Convert the rotated corners to image coordinates using PyTorch operations
    image_corners = torch.stack([torch.tensor(global_to_image_torch(corner, device), dtype=torch.float32, device=device) for corner in rotated_corners])

    # Create a blank mask canvas
    mask = torch.ones((1, image_size[0], image_size[1]), dtype=torch.float, device=device)

    mask = fill_polygon_on_canvas(mask, image_corners)

    print("Mask Shape: ", mask.shape)
    print("Mask: ", mask)

    inverted_mask = 1 - mask

    # Approximate Euclidean Distance Field using PyTorch
    edf = torch.cdist(inverted_mask.unsqueeze(0).unsqueeze(0), inverted_mask.unsqueeze(0).unsqueeze(0)).squeeze()
    print("EDF: ", edf)
    # Normalize the EDF to [0, 1] for consistency in PyTorch
    edf_normalized = (edf - edf.min()) / (edf.max() - edf.min())
    print("EDF normalized: ", edf_normalized)

    edf_normalized = edf_normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Ensure mask also has 4 dimensions before interpolation
    mask = mask.unsqueeze(0)  # Add batch dimension to match interpolation format

    return mask_transformed, edf_transformed


def global_to_image_torch(global_coords, device):
    """
    Convert global coordinates to image coordinates using a differentiable method.

    Args:
    global_coords: Tensor representing global coordinates.

    Returns:
    A tuple of (x_pixel, y_pixel) representing image coordinates.
    """
    # Assuming global_coords is already a tensor on the correct device
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

    adjusted_global_coords = torch.tensor([
        global_coords[1],           # Swap Y to X
        global_coords[0] - 0.5,     # Swap X to Y and adjust
        global_coords[2]            # Z remains the same
    ], dtype=torch.float, device=device)
    
    # Convert adjusted global coordinates to homogeneous coordinates
    global_coords_homogeneous = torch.cat([adjusted_global_coords, torch.tensor([1.0], dtype=torch.float, device=device)])
    
    camera_coords = global_coords_homogeneous[:2]
    camera_coords = torch.cat([camera_coords, torch.tensor([1.0], dtype=torch.float, device=device)])
    image_coords_homogeneous = torch.matmul(K, camera_coords)
    image_coords = image_coords_homogeneous[:2] / image_coords_homogeneous[2]

    # Ensure the coordinates are within image bounds
    x_pixel = torch.clamp(image_coords[0], 0, image_size[1] - 1)
    y_pixel = torch.clamp(image_coords[1], 0, image_size[0] - 1)
    
    return x_pixel, y_pixel

# Example usage:
goal = (((0.478125, 0.04062500000000002, 0.0), (0.0, -0.0, 0.7800583048932599, -0.6257068330832372)), (0.12, 0.12, 0))

# Define the save paths and desired image size
save_path_mask = "goal_mask.png"
save_path_edf = "goal_edf.png"
desired_image_size = (480, 640)

# Create the mask, EDF, and save the images
image_mask, image_edf = create_goal_mask(goal, global_to_image, save_path_mask, save_path_edf, image_size=desired_image_size)

print(f"Mask saved at: {save_path_mask}")
print(f"EDF saved at: {save_path_edf}")

