import numpy as np
import cv2
from pycocotools import mask as mask_utils
from typing import List, Dict, Any,Tuple

def merge_segments(segments: List[Dict], image_shape: Tuple[int, int], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Merges overlapping RLE encoded segments based on IoU for specific labels ('paragraph' and 'text_box').
    Decodes masks into binary, resizes if needed, combines them, and re-encodes them to RLE.

    Args:
        segments (List[Dict]): 
            List of dictionaries, where each dictionary represents a detected segment with keys:
            ['size', 'counts', 'label', 'bbox', 'conf'].
        image_shape (Tuple[int, int]): The shape of the target image (height, width).
        iou_threshold (float): IoU threshold for merging (default=0.5).

    Returns:
        List[Dict]: Filtered list of merged segments.
    """
    height, width = image_shape
    bboxes = np.array([seg['bbox'] for seg in segments])
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    labels = [seg['label'] for seg in segments]

    # Compute pairwise intersection coordinates
    x1 = np.maximum(bboxes[:, None, 0], bboxes[None, :, 0])
    y1 = np.maximum(bboxes[:, None, 1], bboxes[None, :, 1])
    x2 = np.minimum(bboxes[:, None, 2], bboxes[None, :, 2])
    y2 = np.minimum(bboxes[:, None, 3], bboxes[None, :, 3])

    # Compute intersection areas
    intersection_width = np.maximum(0, x2 - x1)
    intersection_height = np.maximum(0, y2 - y1)
    intersection_areas = intersection_width * intersection_height

    # Calculate IoU relative to the smaller bounding box
    smaller_areas = np.minimum(areas[:, None], areas[None, :])
    iou = np.where(smaller_areas > 0, intersection_areas / smaller_areas, 0)

    # Merge segments with IoU > threshold and matching labels
    merged_indices = set()
    updated_segments = []

    for i in range(len(segments)):
        if i in merged_indices:
            continue

        current_segment = segments[i]
        rle_data = {"size": current_segment["size"], "counts": current_segment["counts"]}
        current_mask = mask_utils.decode(rle_data)  # Decode RLE to binary
        if current_mask.shape != (height, width):
            current_mask = cv2.resize(current_mask, (width, height), interpolation=cv2.INTER_NEAREST)  # Resize

        current_bbox = np.array(current_segment['bbox'])

        for j in range(i + 1, len(segments)):
            if j in merged_indices:
                continue

            if iou[i, j] > iou_threshold and labels[i] in {'paragraph', 'text_box'} and labels[j] in {'paragraph', 'text_box'}:
                # Decode and resize the second mask
                rle_data = {"size": segments[j]["size"], "counts": segments[j]["counts"]}
                second_mask = mask_utils.decode(rle_data)
                if second_mask.shape != (height, width):
                    second_mask = cv2.resize(second_mask, (width, height), interpolation=cv2.INTER_NEAREST)

                # Combine the masks (logical OR)
                combined_mask = np.clip(current_mask + second_mask, 0, 1).astype(np.uint8)

                # Update bounding box
                x_min = min(current_bbox[0], bboxes[j, 0])
                y_min = min(current_bbox[1], bboxes[j, 1])
                x_max = max(current_bbox[2], bboxes[j, 2])
                y_max = max(current_bbox[3], bboxes[j, 3])
                merged_bbox = [x_min, y_min, x_max, y_max]

                # Update current segment details
                current_mask = combined_mask
                current_bbox = merged_bbox

                # Mark segment as merged
                merged_indices.add(j)

        # Re-encode the combined mask to RLE
        rle_mask = mask_utils.encode(np.asfortranarray(current_mask))

        # Add the final merged segment to the updated list
        updated_segments.append({
            'size': [height, width],
            'counts': rle_mask['counts'],
            'label': current_segment['label'],
            'bbox': current_bbox,
            'conf': current_segment['conf']
        })

    return updated_segments
