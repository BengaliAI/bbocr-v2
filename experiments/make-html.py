#!/usr/bin/env python
# coding: utf-8


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
from tqdm.auto import tqdm
from utils import *
from apsisocr import ApsisBNOCR
from apsisocr.utils import download
from ultralytics import YOLO
from pathlib import Path
from shapely.geometry import box
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#-----------------------------
# models
#-----------------------------

# YOLO
YOLO_DLA_GID="1n-XbOwUwgMjaFPFzEJ59Avrl9Nc5xsx8"  # Google Drive Link of Yolo Model Weights
# local weight file path
yolo_model_weight_path = "weights/best.pt"
# download if not found
if not os.path.isfile(yolo_model_weight_path ):
    download(YOLO_DLA_GID,yolo_model_weight_path )
# layout analysis model YOLO
model_dla = YOLO(yolo_model_weight_path)

# OCR
model_ocr=ApsisBNOCR()


def run_inference(image_path, file_name, img_src_save_directory):
    """
    This function runs the inference
    Args:
    image_path: Path of the image
    file_name: Name of the file
    img_src_save_directory: Directory to save the image
    Returns:
    region_of_interests: Region of interests
    """
    file_name, extension = file_name.split(".")
    image = cv2.imread(image_path)
    res = model_dla(image)[0]
    res.save(f"reconstruction/image/{file_name}.{extension}")
    region_of_interests = []
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
            x_min, y_min, x_max, y_max = get_original_coordinates(
                normalized_coordinates, info_dict["img_width"], info_dict["img_height"]
            )
            cropped_image_region = image[y_min:y_max, x_min:x_max]
            src = f"{img_src_save_directory}\\{file_name}_{i}.{extension}"
            info_dict["img_src"] = src
            cv2.imwrite(
                f"reconstruction/image/{file_name}_{i}.{extension}",
                cropped_image_region,
            )
        region_of_interests.append(info_dict)
    discard_elements = []
    for i, element in enumerate(tqdm(region_of_interests)):
        bb1 = box(
            element["coordinates"][0],
            element["coordinates"][1],
            element["coordinates"][2],
            element["coordinates"][3],
        )

        for j, other_element in enumerate(region_of_interests):
            if j > i:
                bb2 = box(
                    other_element["coordinates"][0],
                    other_element["coordinates"][1],
                    other_element["coordinates"][2],
                    other_element["coordinates"][3],
                )
                intersection = bb1.intersection(bb2).area
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
    items_deleted = 0
    for index in discard_elements:
        del region_of_interests[index - items_deleted]
        items_deleted += 1
    return region_of_interests


def reconstruct(directory, img_src_save_dir):
    """
    This function reconstructs the image
    Args:
    directory: Directory containing the images
    img_src_save_dir: Directory to save the image
    """

    for file_name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file_name)):

            file_path = directory + "/" + file_name

            print(
                "----------------------------------------------------------------------------"
            )
            print("File name:", file_name)

            start_time = time.time()
            roi = run_inference(
                file_path, file_name, img_src_save_dir
            )
            print(
                "Execution Time for Layout Prediction and Text Recognition:",
                round(time.time() - start_time, 2),
                "seconds",
            )

            start_time = time.time()
            print(roi)
            generate_html(roi, img_src_save_dir, file_name)
            print(
                "Execution Time for Reconstruction:",
                round(time.time() - start_time, 2),
                "seconds",
            )

########## CONSTANTS ############

test_image_directory = "image/"
img_src_save_dir = "image/"

reconstruct(test_image_directory, img_src_save_dir)


