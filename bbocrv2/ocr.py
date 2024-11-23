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
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Union,List,Dict

from apsisocr.apsisnet import ApsisNet
from apsisocr.paddledbnet import PaddleDBNet
from apsisocr.utils import download,LOG_INFO

from ultralytics import YOLO
from pycocotools import mask as cocomask

from .rotation import auto_correct_image_orientation
from .serialization import assign_line_word_numbers

class ImageOCR(object):
    def __init__(self, 
                 yolo_dla_model_path: str = "weights/best.pt", 
                 yolo_dla_model_gid: str = "1n-XbOwUwgMjaFPFzEJ59Avrl9Nc5xsx8",
                 yolo_conf_thresh : float =0.4) -> None:
        """
        Initializes the ImageOCR class by loading the YOLO and OCR models.

        Args:
            yolo_dla_model_path (str): Path to the YOLO model weights.
            yolo_dla_model_gid (str): Google Drive ID to download the YOLO model if not found locally.
            yolo_conf_thresh (float) : The confidence threshold for yolo inference
        """
        # download if not found
        if not os.path.isfile(yolo_dla_model_path):
            download(yolo_dla_model_gid,yolo_dla_model_path)
        self.yolo_conf_thresh=yolo_conf_thresh
        self.dla= YOLO(yolo_dla_model_path)
        LOG_INFO("Loaded Document Layout Analysis Model: Yolov8")
        self.bn_rec=ApsisNet()
        LOG_INFO("Loaded Bangla Recognition Model: ApsisNet")
        self.detector=PaddleDBNet(load_line_model=False)        
        LOG_INFO("Loaded Word detector Model: Paddle-DBnet")
    
    def process_ocr(self,image:np.ndarray)->Tuple[List[dict],dict]:
        """
            Args:
                image(np.ndarray) : Input Image
            Returns:
                Tuple[List[dict],dict] 
                    - List[dict]: words   - OCR recognition results as words.
                                          - keys for each word ['poly','text',"line_num","word_num"]
                    - dict : rotation_data- orientation information for ocr
                                          - keys for meta_data ["rotated_image","rotated_mask","angle"]   
        """
        # extract word regions
        regions=self.detector.get_word_boxes(image)
        # correct for rotation
        rotated_image,rotated_mask,angle,rotated_polys=auto_correct_image_orientation(image,regions)
        # get word crops
        crops=self.detector.get_crops(rotated_image,rotated_polys)
        # get text
        texts=self.bn_rec.infer(crops)
        # get word and line number
        words=[{"poly":poly,"text":txt} for poly,txt in zip(rotated_polys,texts)]
        words=assign_line_word_numbers(words)
        # format output data
        rotation_data={ "rotated_image":rotated_image,
                        "rotated_mask":rotated_mask,
                        "angle":angle}
        
        return words,rotation_data



    def process_dla(self,image:np.ndarray)->list[dict]:
        """
            Process Document Layout analysis data. 
            Masks,BBoxes,Lables and confidence socres are processed in this function. 
            - labels are converted to class names
            - Masks are encoded with RLE

            Args:
                image(np.ndarray) : Input Image
            
            Returns: 
                list[dict]: segments- YOLO detection results as list of RLE encoded segments.
                                    - keys for each segment ['size', 'counts', 'label', 'bbox','conf']

        """
        # Perform YOLO object detection
        res = self.dla(image,conf=self.yolo_conf_thresh)[0]
        
        segments = []

        # Extract masks, labels, and bounding boxes from the results
        masks  = res.masks.data.cpu().numpy()  # Get masks as a NumPy array
        labels = [res.names[int(i)] for i in res.boxes.cls.cpu().numpy()]  # Get labels corresponding to detected classes
        bboxes = res.boxes.data.cpu().numpy()[:, :4].astype(int)  # Get bounding boxes and convert to integer format
        confs  = res.boxes.conf.cpu().numpy()
        # Iterate through bounding boxes, labels, and masks
        for bbox, label, mask,conf in zip(bboxes, labels, masks,confs):
            # Encode the binary mask into COCO RLE format
            segment = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
            
            # Add additional information to the segment (label and bounding box)
            segment["label"] = label
            segment["bbox"]  = bbox
            segment["conf"]  = conf

            # Append the segment to the list of segments
            segments.append(segment)
        return segments

    def __call__(self, image: Union[str, np.ndarray]) -> dict:
        """
        Processes an image with YOLO for object detection and OCR for text recognition.

        Args:
            image (Union[str, np.ndarray]): Input image. Can be a file path or a NumPy array.

        Returns:
            dict : that holds two keys: ["words","segments","rotation"]
                    segments- YOLO detection results as list of RLE encoded segments.
                            - keys for each segment ['size', 'counts', 'label', 'bbox','conf']
                    words   - OCR recognition results as words.
                            - keys for each word ['poly','text','line_num','word_num']
                    rotation- orientation information for ocr
                            - keys for meta_data ["rotated_image","rotated_mask","angle"]   
                    
        """
        # If the image is a file path, read and convert it.
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        # Perform OCR on the image
        words,rotation_data= self.process_ocr(image)
        # dla segmentation
        segments=self.process_dla(rotation_data["rotated_image"])
        
        return {"words":words,"segments":segments,"rotation":rotation_data}

