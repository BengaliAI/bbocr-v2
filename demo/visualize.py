import cv2
import numpy as np
from typing import List, Dict, Any
from pycocotools import mask as cocomask

def draw_word_polys(image: np.ndarray, output: Dict[str, List[Dict[str, Any]]]) -> np.ndarray:
    """
    Draws the word polygons from OCR results on the input image.
    
    Parameters:
        image (np.ndarray): The input image.
        output (Dict[str, List[Dict[str, Any]]]): A dictionary containing "words" and "segments".
            - "words" is a list of OCR recognition results with keys ['poly'].

    Returns:
        np.ndarray: The image with drawn word polygons.
    """
    # Make a copy of the image to draw on
    output_image = image.copy()
    
    # Iterate over the words
    for word in output.get("words", []):
        poly = np.array(word['poly'], dtype=np.int32)  # Convert polygon to NumPy array
        
        # Draw the polygon
        cv2.polylines(output_image, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    return output_image


def draw_masks_with_labels(
    image: np.ndarray,
    output: Dict[str, List[Dict[str, Any]]],
    transparency: float = 0.5
) -> np.ndarray:
    """
    Draws masks, labels, and bounding boxes on the image with unique colors for each label.
    Adds a white padding at the top to display label names and their colors.
    
    Parameters:
        image (np.ndarray): The input image.
        output (Dict[str, List[Dict[str, Any]]]): A dictionary containing "segments".
            - Each segment includes 'size', 'counts', 'label', and 'bbox'.
        transparency (float): Transparency factor for masks (0: fully transparent, 1: fully opaque).
    
    Returns:
        np.ndarray: The image with drawn masks, labels, and bounding boxes.
    """
    # Define colors for the labels
    label_colors = {
        "paragraph": (255, 0, 0),     # Blue
        "text_box": (0, 255, 0),     # Green
        "image": (0, 0, 255),        # Red
        "table": (255, 255, 0),      # Cyan
    }

    # Add white padding to the top of the image for displaying labels
    padding_height = 50  # Height of the padding
    img_height, img_width = image.shape[:2]
    padded_image = np.ones((img_height + padding_height, img_width, 3), dtype=np.uint8) * 255
    padded_image[padding_height:] = image  # Add the original image below the padding

    overlay = padded_image.copy()

    # Draw masks and bounding boxes
    for segment in output.get("segments", []):
        # Decode mask using RLE
        rle = {"size": segment["size"], "counts": segment["counts"]}
        mask = cocomask.decode(rle)  # Decode RLE to binary mask
        
        # Resize mask to match the image size
        mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        # Get the label and corresponding color
        label = segment["label"]
        color = label_colors.get(label, (255, 255, 255))  # Default to white if label is unknown
        
        # Apply mask with transparency
        for c in range(3):  # Apply color to all channels
            overlay[padding_height:, :, c] = np.where(
                mask_resized == 1,
                (1 - transparency) * overlay[padding_height:, :, c] + transparency * color[c],
                overlay[padding_height:, :, c]
            )

        # Draw bounding box
        bbox = segment["bbox"]  # Format: [x_min, y_min, x_max, y_max]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1 + padding_height), (x2, y2 + padding_height), color, thickness=1)

    # Draw label names and their corresponding colors in the padding
    for i, (label, color) in enumerate(label_colors.items()):
        cv2.rectangle(overlay, (10 + i * 100, 10), (90 + i * 100, 40), color, -1)  # Color block
        cv2.putText(overlay, label, (10 + i * 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return overlay
