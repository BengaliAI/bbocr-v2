from typing import List, Dict, Union, Any
import numpy as np

def calculate_bbox_from_poly(poly: List[int]) -> List[int]:
    """
    Convert a polygon into its bounding box representation.

    Args:
        poly (List[int]): A list of integers representing the vertices of a polygon. 
                          Format: [x1, y1, x2, y2, ..., xn, yn].

    Returns:
        List[int]: A bounding box in the format [x_min, y_min, x_max, y_max], 
                   where:
                   - x_min is the smallest x-coordinate of the polygon.
                   - y_min is the smallest y-coordinate of the polygon.
                   - x_max is the largest x-coordinate of the polygon.
                   - y_max is the largest y-coordinate of the polygon.

    Example:
        Input: [10, 20, 30, 40, 50, 60, 70, 80]
        Output: [10, 20, 70, 80]
    """
    xs = poly[0::2]  # Extract x-coordinates
    ys = poly[1::2]  # Extract y-coordinates
    return [min(xs), min(ys), max(xs), max(ys)]


def calculate_intersection_area(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the intersection area between two axis-aligned bounding boxes.

    Args:
        bbox1 (np.ndarray): A NumPy array representing the first bounding box 
                            in the format [x_min, y_min, x_max, y_max].
        bbox2 (np.ndarray): A NumPy array representing the second bounding box 
                            in the format [x_min, y_min, x_max, y_max].

    Returns:
        float: The area of the intersection between the two bounding boxes. 
               If the boxes do not intersect, the result is 0.0.

    Calculation:
        - Find the overlapping region between the two boxes by computing:
            - The maximum of the minimum x-coordinates (`x_min`).
            - The maximum of the minimum y-coordinates (`y_min`).
            - The minimum of the maximum x-coordinates (`x_max`).
            - The minimum of the maximum y-coordinates (`y_max`).
        - Compute the width and height of the overlapping region.
        - Return the product of width and height as the intersection area.

    Example:
        Input: bbox1 = [10, 10, 50, 50], bbox2 = [30, 30, 70, 70]
        Output: 400.0 (the intersection area is a square 20x20)

    Note:
        The function assumes axis-aligned bounding boxes (no rotation).
    """
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])

    # Calculate width and height of the intersection
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)

    return width * height

