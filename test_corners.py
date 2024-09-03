import torch
import numpy as np
from PIL import Image

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
    canvas[0] = mask

    return canvas


# Example usage with provided tensor
canvas = torch.ones((1, 369, 492), dtype=torch.float32, device='cuda:0')
image_corners = torch.tensor([
    [284.8192, 161.1746],
    [244.2957, 152.1671],
    [235.2882, 192.6906],
    [275.8117, 201.6981]
], device='cuda:0')

# Fill the canvas using the polygon defined by image_corners
canvas = fill_polygon_on_canvas(canvas, image_corners)

# Move the tensor to CPU and convert to NumPy array
canvas_np = canvas.squeeze(0).cpu().numpy()  # Remove the batch dimension

# Normalize the canvas to range [0, 255] for image saving
canvas_np = (canvas_np * 255).astype(np.uint8)

# Convert the NumPy array to a PIL image
canvas_image = Image.fromarray(canvas_np)

# Save the image
canvas_image.save("filled_polygon.png")
print("Image saved as filled_polygon.png")