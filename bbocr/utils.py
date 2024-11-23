import math
import copy
import numpy as np
import os

from typing import List, Dict, Tuple
from .htmlgenerator import HtmlGenerator,INFO_DICT,names

def generate_html(detected_elements_info: List[Dict], save_path: str,template_dir :str) -> None:
    """
    Generates an HTML file based on detected elements information.

    Args:
        detected_elements_info (List[Dict]): A list of dictionaries containing information about detected elements.
        save_path (str): Name of the generated HTML file
        template_dir (str) : Directory for storing templates
    """
    file_name=os.path.basename(save_path)
    gen = HtmlGenerator(file_name,template_dir)  # Initialize HTML generator with the base file name.

    for element_info in detected_elements_info:
        # Handle different detected element classes and insert corresponding HTML elements.
        if element_info["class"] == "paragraph":
            gen.insert_paragraph(element_info)

        elif element_info["class"] == "text_box":
            gen.insert_text_box(element_info)

        elif element_info["class"] == "image":
            gen.insert_image(element_info)

    # Create the HTML file in the specified directory.
    gen.create_html_file(save_path)
    return gen.get_html_data()


def get_normalized_coordinates(xyxy_tensor: np.ndarray, height: int, width: int) -> List[float]:
    """
    Converts bounding box coordinates into normalized format.

    Args:
        xyxy_tensor (np.ndarray): Tensor containing bounding box coordinates in [x_min, y_min, x_max, y_max] format.
        height (int): Height of the image.
        width (int): Width of the image.

    Returns:
        List[float]: Normalized coordinates in [x_min, y_min, x_max, y_max] format.
    """
    x_min = xyxy_tensor[0][0].item() / width
    y_min = xyxy_tensor[0][1].item() / height
    x_max = xyxy_tensor[0][2].item() / width
    y_max = xyxy_tensor[0][3].item() / height

    return [x_min, y_min, x_max, y_max]


def get_original_coordinates(normalized_coordinates: List[float], image_width: int, image_height: int) -> List[int]:
    """
    Converts normalized coordinates back to original image scale.

    Args:
        normalized_coordinates (List[float]): Normalized coordinates in [x_min, y_min, x_max, y_max] format.
        image_width (int): Width of the original image.
        image_height (int): Height of the original image.

    Returns:
        List[int]: Original coordinates in [x_min, y_min, x_max, y_max] format.
    """
    orig_coordinates = [None] * 4

    orig_coordinates[0] = math.floor(normalized_coordinates[0] * image_width)
    orig_coordinates[1] = math.floor(normalized_coordinates[1] * image_height)
    orig_coordinates[2] = math.ceil(normalized_coordinates[2] * image_width)
    orig_coordinates[3] = math.ceil(normalized_coordinates[3] * image_height)

    return orig_coordinates


def top_bottom_padding(cropped_text_region: np.ndarray) -> np.ndarray:
    """
    Adds top and bottom padding to a cropped text region.

    Args:
        cropped_text_region (np.ndarray): Image region of the cropped text.

    Returns:
        np.ndarray: Padded image with white background.
    """
    h, w = cropped_text_region.shape[:2]
    padded_height = int(h * 1.5)
    padded_width = w

    # Create a white background for padding.
    padded_image = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * 255

    # Calculate top and bottom padding positions.
    top_padding = (h * 2 - h) // 2
    bottom_padding = top_padding + h

    # Place the cropped text region in the center of the padded area.
    padded_image[top_padding:bottom_padding, :] = cropped_text_region

    return padded_image


def process_html_block(data_box) -> Tuple[Dict, List[float]]:
    """
    Processes a single data box to extract its information for HTML generation.

    Args:
        data_box: Object containing bounding box data, class information, and original image dimensions.

    Returns:
        Tuple[Dict, List[float]]: 
            - A dictionary containing processed information for HTML generation.
            - Normalized coordinates in [x_min, y_min, x_max, y_max] format.
    """
    # Create a deep copy of the template info dictionary.
    info_dict = copy.deepcopy(INFO_DICT)

    # Extract class ID and original shape of the image.
    cls = data_box.cls.item()
    img_height, img_width = data_box.orig_shape

    # Compute normalized coordinates.
    normalized_coordinates = get_normalized_coordinates(
        data_box.xyxy, img_height, img_width
    )

    # Assign class name based on class ID.
    if cls == 0:
        info_dict["class"] = names[0]
    elif cls == 1:
        info_dict["class"] = names[1]
    elif cls == 2:
        info_dict["class"] = names[2]
    elif cls == 3:
        info_dict["class"] = names[3]

    # Populate additional information for the HTML block.
    info_dict["coordinates"] = normalized_coordinates
    info_dict["left"], info_dict["top"] = (
        normalized_coordinates[0] * 100,
        normalized_coordinates[1] * 100,
    )
    info_dict["img_height"], info_dict["img_width"] = img_height, img_width
    info_dict["elem_width"] = (normalized_coordinates[2] - normalized_coordinates[0]) * 100
    info_dict["elem_height"] = (normalized_coordinates[3] - normalized_coordinates[1]) * 100

    return info_dict, normalized_coordinates
