#--------------------------------
# imports
#--------------------------------

import os
import cv2
import time
import math
import numpy as np
import gdown
import copy
from typing import List, Dict
from shapely.geometry import box
from tqdm import tqdm

from .utils import process_html_block , get_original_coordinates, top_bottom_padding
from apsisocr import ApsisBNOCR
from apsisocr.utils import download
from ultralytics import YOLO
from pathlib import Path
from shapely.geometry import box


def run_inference(image_path : str, 
                  img_src_save_directory : str,
                  model_dla:YOLO,
                  model_ocr:ApsisBNOCR) -> list[dict]:
    """
    This function runs the inference
    
    Args:
        image_path (str): Path of the image
        img_src_save_directory (str): Directory to save the image
        model_dla (YOLO) : a yolo model for Document layout analysis
        model_ocr (ApsisBNOCR) : a ApsisNet-PaddleDBNET model for ocr 
    Returns:
        region_of_interests (list[dict]): Region of interests
    """
    region_of_interests = []
    
    filename=os.path.basename(image_path)
    
    extension=filename.split(".")[-1]
    basename=filename.split(".")[0]
    
    image = cv2.imread(image_path)
    
    res = model_dla(image)[0]
    
    print("processing dla outputs")
    for i in tqdm(range(len(res.boxes))):
        
        info_dict, normalized_coordinates = process_html_block(res.boxes[i])
        
        # do ocr if text or else save image
        if info_dict["class"] == "paragraph" or info_dict["class"] == "text_box":

            x_min, y_min, x_max, y_max = get_original_coordinates(normalized_coordinates, info_dict["img_width"], info_dict["img_height"])
            
            cropped_text_region = image[y_min:y_max, x_min:x_max]
            
            cropped_text_region = top_bottom_padding(cropped_text_region)
            
            ocr_result = model_ocr(cropped_text_region)
            
            text = ocr_result["text"].replace("\n","")
            
            info_dict["text"] = text
        
        elif info_dict["class"] == "image":
            x_min, y_min, x_max, y_max = get_original_coordinates(normalized_coordinates, info_dict["img_width"], info_dict["img_height"])

            cropped_image_region = image[y_min:y_max, x_min:x_max]
            
            src = os.path.join(img_src_save_directory,f"{basename}_{i}.{extension}")
            
            info_dict["img_src"] = src
            
            cv2.imwrite(src,cropped_image_region)

        region_of_interests.append(info_dict)
    
    return region_of_interests


def clean_region_of_interests(region_of_interests: List[Dict]) -> List[Dict]:
    """
    Removes overlapping bounding boxes from a list of region of interests (ROIs) based on Intersection over Union (IoU).
    If two bounding boxes overlap with IoU > 0.5, the smaller bounding box is discarded.

    Args:
        region_of_interests (List[Dict]): 
            A list of dictionaries, where each dictionary represents a region of interest (ROI) 
            with coordinates in the format {"coordinates": [x_min, y_min, x_max, y_max]}.

    Returns:
        List[Dict]: The filtered list of ROIs with overlapping boxes removed.
    """
    discard_elements = []
    
    for i, element in enumerate(tqdm(region_of_interests, desc="Processing ROIs")):
        # Create a bounding box for the current ROI
        bb1 = box(*element["coordinates"])

        for j, other_element in enumerate(region_of_interests):
            if j > i:  # Avoid redundant comparisons
                # Create a bounding box for the other ROI
                bb2 = box(*other_element["coordinates"])
                
                # Calculate intersection area
                intersection = bb1.intersection(bb2).area
                
                # Determine IoU based on the smaller bounding box
                if bb1.area < bb2.area:
                    iou = intersection / bb1.area
                    if iou > 0.5:
                        if i not in discard_elements:
                            discard_elements.append(i)
                else:
                    iou = intersection / bb2.area
                    if iou > 0.5:
                        if j not in discard_elements:
                            discard_elements.append(j)

    # Remove ROIs marked for discard
    items_deleted = 0
    for index in discard_elements:
        del region_of_interests[index - items_deleted]
        items_deleted += 1

    return region_of_interests


def clean_region_of_interests_vectorized(region_of_interests: List[Dict]) -> List[Dict]:
    """
    Vectorized implementation to remove overlapping bounding boxes from a list of ROIs based on IoU.
    If two bounding boxes overlap with IoU > 0.5, the smaller bounding box is discarded.

    Args:
        region_of_interests (List[Dict]): 
            A list of dictionaries, where each dictionary represents a region of interest (ROI) 
            with coordinates in the format {"coordinates": [x_min, y_min, x_max, y_max]}.

    Returns:
        List[Dict]: The filtered list of ROIs with overlapping boxes removed.
    """
    # Convert bounding box coordinates into a NumPy array
    coordinates = np.array([roi["coordinates"] for roi in region_of_interests])

    # Precompute areas of all bounding boxes
    areas = (coordinates[:, 2] - coordinates[:, 0]) * (coordinates[:, 3] - coordinates[:, 1])

    # Compute pairwise intersection coordinates
    x1 = np.maximum(coordinates[:, None, 0], coordinates[None, :, 0])
    y1 = np.maximum(coordinates[:, None, 1], coordinates[None, :, 1])
    x2 = np.minimum(coordinates[:, None, 2], coordinates[None, :, 2])
    y2 = np.minimum(coordinates[:, None, 3], coordinates[None, :, 3])

    # Compute intersection areas
    intersection_width = np.maximum(0, x2 - x1)
    intersection_height = np.maximum(0, y2 - y1)
    intersection_areas = intersection_width * intersection_height

    # Calculate IoU relative to the smaller bounding box
    smaller_areas = np.minimum(areas[:, None], areas[None, :])
    iou = np.where(smaller_areas > 0, intersection_areas / smaller_areas, 0)

    # Ignore self-comparisons and redundant calculations
    iou[np.triu_indices_from(iou)] = 0  # Zero out upper triangle and diagonal
    overlaps = np.argwhere(iou > 0.5)  # Find all pairs with IoU > 0.5

    # Determine which indices to discard
    discard_indices = set()
    for i, j in overlaps:
        # Discard the smaller box; if equal, arbitrarily discard one
        if areas[i] < areas[j]:
            discard_indices.add(i)
        else:
            discard_indices.add(j)

    # Filter out discarded indices
    filtered_region_of_interests = [
        roi for idx, roi in enumerate(region_of_interests) if idx not in discard_indices
    ]

    return filtered_region_of_interests
