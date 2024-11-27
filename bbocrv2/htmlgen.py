import numpy as np
from typing import List, Dict, Any
from PIL import Image  # Import Image class from PIL
from io import BytesIO  # Import BytesIO for in-memory image buffering
import base64  # Import base64 for encoding image to base64

def sort_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort segments by their position on the page.

    Args:
        segments (List[Dict[str, Any]]): A list of segments, where each segment contains:
            - 'bbox': The bounding box of the segment in [x_min, y_min, x_max, y_max] format.

    Returns:
        List[Dict[str, Any]]: The list of segments sorted by the y-axis first (top to bottom), 
                              then by the x-axis (left to right).
    """
    return sorted(segments, key=lambda seg: (seg["bbox"][1], seg["bbox"][0]))


# def generate_html(segments_data: List[Dict[str, Any]], page_width: int, page_height: int, img2) -> str:
#     """
#     Generate an HTML representation of segments data, maintaining relative positions and alignments.

#     Args:
#         segments_data (List[Dict[str, Any]]): 
#             The output of `process_segments_and_words`, where each segment contains:
#             - 'label': The type of the segment ('paragraph', 'text_box', etc.).
#             - 'bbox': The bounding box of the segment in [x_min, y_min, x_max, y_max] format.
#             - 'data': None for 'image'/'table', or a list of word details for 'paragraph'/'text_box':
#                 - 'relative_line_num': Line number relative to the segment.
#                 - 'relative_word_num': Word number relative to the line.
#                 - 'text': The word text.
#                 - 'poly': The word's polygon coordinates.
#         page_width (int): The width of the page or canvas for scaling purposes.
#         page_height (int): The height of the page or canvas for scaling purposes.

#     Returns:
#         str: An HTML string representing the segments with relative positioning and alignment.
#     """
#     def calculate_bbox_from_poly(poly: List[int]) -> List[int]:
#         """Convert a polygon into a bounding box [x_min, y_min, x_max, y_max]."""
#         xs = poly[0::2]
#         ys = poly[1::2]
#         return [min(xs), min(ys), max(xs), max(ys)]

#     # Start HTML structure
#     html = [
#         f"<div style='position:relative; width:{page_width}px; height:{page_height}px; border:1px solid black;'>"
#     ]

#     for segment in segments_data:
#         if segment["label"] in ["paragraph", "text_box"] and segment["data"]:
#             # Extract bounding box for the segment
#             seg_bbox = segment["bbox"]
#             seg_style = (
#                 f"position:absolute; left:{seg_bbox[0]}px; top:{seg_bbox[1]}px; "
#                 f"width:{seg_bbox[2] - seg_bbox[0]}px; height:{seg_bbox[3] - seg_bbox[1]}px; overflow:hidden;"
#             )
#             html.append(f"<div style='{seg_style}'>")

#             # Organize words into lines based on relative_line_num
#             lines = {}
#             for word in segment["data"]:
#                 line_num = word["relative_line_num"]
#                 if line_num not in lines:
#                     lines[line_num] = []
#                 lines[line_num].append(word)

#             # Sort lines and words
#             for line_num in sorted(lines.keys()):
#                 line_words = sorted(lines[line_num], key=lambda w: w["relative_word_num"])
#                 html.append("<div style='position:relative; display:block; white-space:nowrap;'>")

#                 for word in line_words:
#                     word_bbox = calculate_bbox_from_poly(word["poly"])
#                     word_width = word_bbox[2] - word_bbox[0]
#                     word_height = word_bbox[3] - word_bbox[1]
#                     word_style = (
#                         f"position:relative; display:inline-block; "
#                         f"width:{word_width}px; height:{word_height}px; "
#                         f"vertical-align:top; margin:0; padding:0;"
#                     )
#                     html.append(f"<span style='{word_style}'>{word['text']}</span>")

#                 html.append("</div>")  # Close line div

#             html.append("</div>")  # Close segment div

#         elif segment["label"] == "image" and segment["bbox"] is not None:
#             # Process image segments
#             x_min, y_min, x_max, y_max = map(int, segment["bbox"])
#             img2 = np.asarray(img2)  # Ensure the image is a NumPy array
#             sliced_image = img2[y_min:y_max, x_min:x_max]  # Slice the image

#             # Convert the NumPy array to a PIL Image
#             pil_image = Image.fromarray(sliced_image)

#             # Save the image to an in-memory buffer
#             buffer = BytesIO()
#             pil_image.save(buffer, format="PNG")
#             buffer.seek(0)

#             # Encode the buffer as Base64
#             img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

#             # Add the Base64 image to HTML
#             html.append(
#                 f"<img src='data:image/png;base64,{img_base64}' "
#                 f"style='{page_width}; {page_height}; display:block; margin:auto;' alt='Image'>"
#             ) 


#     html.append("</div>")  # Close page div

#     return "\n".join(html)

def generate_html(segments_data: List[Dict[str, Any]], page_width: int, page_height: int, img2) -> str:
    """
    Generate an HTML representation of segments data, with each segment placed in a paragraph
    and images handled separately.
    """

    def calculate_bbox_from_poly(poly: List[int]) -> List[int]:
        """Convert a polygon into a bounding box [x_min, y_min, x_max, y_max]."""
        xs = poly[0::2]
        ys = poly[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]

    # Start HTML structure
    html = [
        # f"<div style='position:relative; width:{page_width}px; height:{page_height}px;'>"
        f"<div style='position:relative; width:{page_width}px; height:{page_height}px; background-color:white;'>"
    ]

    segments_data = sort_segments(segments_data)
    for segment in segments_data:
        if segment["label"] in ["paragraph", "text_box"] and segment["data"]:
            # Start a new paragraph for each segment
            html.append("<p style='margin:10px 0; background-color:transparent; color:black; font-size:14px;'>")

            # Organize text by lines
            lines = {}
            for word in segment["data"]:
                line_num = word["relative_line_num"]
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(word)

            # Append each line of text
            for line_num in sorted(lines.keys()):
                line_words = sorted(lines[line_num], key=lambda w: w["relative_word_num"])
                for word in line_words:
                    html.append(f"{word['text']} ")  # Add word with space
                html.append("<br>")  # Line break

            html.append("</p>")  # Close paragraph

        elif segment["label"] == "image" and segment["bbox"] is not None:
            # Handle image segments with bounding box
            x_min, y_min, x_max, y_max = map(int, segment["bbox"])
            img2 = np.asarray(img2)  # Ensure the image is a NumPy array
            sliced_image = img2[y_min:y_max, x_min:x_max]  # Slice the region

            # Convert to PIL Image
            pil_image = Image.fromarray(sliced_image)

            # Save to buffer and encode as Base64
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            # Add the sliced image as an inline image element
            html.append(
                f"<div style='margin:10px 0;'>"
                f"<img src='data:image/png;base64,{img_base64}' "
                f"style='display:block; max-width:100%; height:auto;' alt='Segment Image'>"
                f"</div>"
            )

    html.append("</div>")  # Close container div

    return "\n".join(html)

