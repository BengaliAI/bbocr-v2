import numpy as np
import cv2
from statistics import median_low
from typing import List, Tuple


def rotate_image(mat: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image by a given angle in degrees, expanding its bounds to avoid cropping.
    
    Args:
        mat: The input image as a numpy array.
        angle: The angle in degrees by which the image will be rotated.

    Returns:
        Rotated image as a numpy array with adjusted bounds.
    """
    # Get the height and width of the image
    height, width = mat.shape[:2]
    # Compute the image center
    image_center = (width / 2, height / 2)

    # Compute the rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Calculate the new image bounds
    abs_cos = abs(rotation_mat[0, 0])  # Cosine of rotation angle
    abs_sin = abs(rotation_mat[0, 1])  # Sine of rotation angle
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # Adjust the translation component of the rotation matrix
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # Apply the affine transformation to rotate the image
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST)
    return rotated_mat

def create_mask(image: np.ndarray, regions: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    Create a mask from a list of polygonal regions.

    Args:
        image: The input image as a numpy array.
        regions: A list of regions, where each region is a list of (x, y) coordinates.

    Returns:
        A mask with the same height and width as the input image. Each region is filled with unique values.
    """
    # Get the height and width of the image
    h, w, _ = image.shape
    # Initialize a blank mask
    mask = np.zeros((h, w))
    for i, region in enumerate(regions):
        # Convert the region coordinates to integer format and reshape as needed
        region = np.array(region).astype(np.int32).reshape(-1, 2)
        # Fill the region on the mask with a unique value (i + 1)
        cv2.fillPoly(mask, [region.reshape((-1, 1, 2))], i + 1)
    return mask


def auto_correct_image_orientation(
    image: np.ndarray, regions: List[List[Tuple[int, int]]]
) -> Tuple[np.ndarray, np.ndarray, int, List[List[Tuple[int, int]]]]:
    """
    Automatically correct the orientation of an image based on detected contours in a mask.

    Args:
        image: The input image as a numpy array.
        regions: A list of regions, where each region is a list of (x, y) coordinates.

    Returns:
        A tuple containing:
        - The rotated image.
        - The rotated mask.
        - The applied rotation angle (in degrees).
        - The rotated polygons (list of transformed regions).
    """
    # Create mask from regions
    mask = create_mask(image, regions)

    # Calculate the angles of all regions
    angles = []
    for region in regions:
        poly = np.array(region).astype(np.int32).reshape(-1, 2)
        bottom_left = poly[3]  # Bottom-left corner
        bottom_right = poly[2]  # Bottom-right corner
        angle = np.arctan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0])
        angles.append(np.degrees(angle))

    # Use the median angle to minimize the effect of outliers
    angle = int(median_low(angles))

    # Rotate the image
    rotated_image = rotate_image(image, angle)
    rotated_mask = rotate_image(mask, angle)

    # Get the dimensions of the original image
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Compute the rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Adjust the rotation matrix to match the new bounds
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - center[0]
    rotation_mat[1, 2] += bound_h / 2 - center[1]

    # Rotate the polygons
    rotated_polys = []
    for region in regions:
        poly = np.array(region, dtype=np.float32).reshape(-1, 2)
        # Apply affine transformation to the polygon points
        rotated_poly = cv2.transform(np.array([poly]), rotation_mat)[0]
        # Flatten the polygon points to [x1, y1, x2, y2, ...] format
        rotated_polys.append(rotated_poly.flatten().astype(int).tolist())
    return rotated_image, rotated_mask, angle, rotated_polys
