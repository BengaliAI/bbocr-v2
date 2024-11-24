from typing import List, Dict, Union, Any
import numpy as np
from .utils import calculate_bbox_from_poly,calculate_intersection_area

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


def sort_lines_and_words(data: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Sort lines and words within a segment by relative line and word numbers.

    Args:
        data (List[Dict[str, Any]]): A list of word data, where each word contains:
            - 'relative_line_num': The relative line number in the segment.
            - 'relative_word_num': The relative word number in the line.
            - 'text': The text of the word.

    Returns:
        Dict[int, List[Dict[str, Any]]]: A dictionary where:
            - Keys are the relative line numbers.
            - Values are lists of words sorted by their relative word numbers.
    """
    sorted_lines = {}
    for word in data:
        line_num = word["relative_line_num"]
        if line_num not in sorted_lines:
            sorted_lines[line_num] = []
        sorted_lines[line_num].append(word)

    # Sort words in each line by their relative word number
    for line in sorted_lines.values():
        line.sort(key=lambda w: w["relative_word_num"])

    return sorted_lines

def process_segments_and_words(
    segments: List[Dict[str, Any]],
    words: List[Dict[str, Any]]
) -> List[Dict[str, Union[str, List[Dict[str, Union[int, str, List[int]]]], None]]]:
    """
    Processes YOLO detection results (segments) and OCR results (words) to create structured data 
    with segment-specific information and word details relative to their segment.

    Args:
        segments (List[Dict[str, Any]]): A list of detected segments. Each segment contains:
            - 'label': The type of segment (e.g., paragraph, text_box, table, image).
            - 'bbox': The bounding box of the segment in [x1, y1, x2, y2] format.
            - 'size', 'counts', 'conf' (ignored for this processing).
        
        words (List[Dict[str, Any]]): A list of OCR-recognized words. Each word contains:
            - 'poly': The polygon coordinates of the word in [x1, y1, x2, y2, ...] format.
            - 'text': The text of the word.
            - 'line_num': The absolute line number in the whole document.
            - 'word_num': The absolute word number within the line.

    Returns:
        List[Dict[str, Union[str, List[Dict[str, Union[int, str, List[int]]]], None]]]: 
        A list of dictionaries, each containing:
            - 'label': The type of the segment.
            - 'bbox': The bounding box of the segment.
            - 'data': 
                - None, if the segment is a 'table' or 'image'.
                - A list of word details for 'paragraph' or 'text_box', each containing:
                    - 'relative_line_num': The relative line number within the segment.
                    - 'relative_word_num': The relative word number in the line within the segment.
                    - 'text': The text of the word.
                    - 'poly': The polygon of the word.
    """
    # Precompute bounding boxes for words
    word_bboxes = [calculate_bbox_from_poly(word["poly"]) for word in words]
    segment_bboxes = np.array([seg["bbox"] for seg in segments])

    # Assign words to the segment with the maximum intersection
    word_assignments = []
    for word_idx, word_bbox in enumerate(word_bboxes):
        word_bbox_np = np.array(word_bbox)
        intersections = np.array([
            calculate_intersection_area(word_bbox_np, seg_bbox)
            for seg_bbox in segment_bboxes
        ])

        # Assign the word to the segment with the maximum intersection
        max_idx = np.argmax(intersections)
        word_assignments.append((word_idx, max_idx))

    # Group words by segment
    segment_word_map = {i: [] for i in range(len(segments))}
    for word_idx, segment_idx in word_assignments:
        segment_word_map[segment_idx].append(words[word_idx])

    # Prepare the output
    output = []
    for segment_idx, segment in enumerate(segments):
        label = segment["label"]
        bbox = segment["bbox"]

        # Initialize segment info
        segment_info = {"label": label, "bbox": bbox}

        if label in ["image", "table"]:
            # No data for image or table segments
            segment_info["data"] = None
        else:
            # Process words assigned to this segment
            segment_words = segment_word_map[segment_idx]

            # Group words by their absolute line numbers
            lines = {}
            for word in segment_words:
                line_num = word["line_num"]
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(word)

            # Assign relative line and word numbers
            data = []
            for rel_line_num, (abs_line_num, line_words) in enumerate(sorted(lines.items()), start=1):
                for rel_word_num, word in enumerate(sorted(line_words, key=lambda x: x["word_num"]), start=1):
                    data.append({
                        "relative_line_num": rel_line_num,
                        "relative_word_num": rel_word_num,
                        "text": word["text"],
                        "poly": word["poly"]
                    })

            segment_info["data"] = data

        output.append(segment_info)

    return output

def construct_text_from_segments(segments_data: List[Dict[str, Any]]) -> str:
    """
    Constructs a multiline text from processed segment data, with segment labels and serial numbers.

    Args:
        segments_data (List[Dict[str, Any]]): 
            The output of `process_segments_and_words`, where each segment contains:
            - 'label': The type of the segment ('paragraph', 'text_box', etc.).
            - 'bbox': The bounding box of the segment in [x_min, y_min, x_max, y_max] format.
            - 'data': None for 'image'/'table', or a list of word details for 'paragraph'/'text_box':
                - 'relative_line_num': Line number relative to the segment.
                - 'relative_word_num': Word number relative to the line.
                - 'text': The word text.
                - 'poly': The word's polygon coordinates.

    Returns:
        str: A multiline string where:
             - Each segment starts with its label and serial number (e.g., "paragraph:serial:1").
             - Lines within each segment are joined by '\n'.
             - Words within each line are joined by spaces.
             - Segments are separated by three newline characters.
    """
    text_parts = []

    # Filter and sort segments
    text_segments = [seg for seg in segments_data if seg["label"] in ["paragraph", "text_box"]]
    sorted_segments = sort_segments(text_segments)

    for idx, segment in enumerate(sorted_segments, start=1):
        label = segment["label"]
        segment_header = f"{label}:serial:{idx}"
        if segment["data"]:
            # Group and sort words into lines
            sorted_lines = sort_lines_and_words(segment["data"])

            # Construct the text for the segment
            segment_text = "\n".join(
                " ".join(word["text"] for word in line) for line in sorted_lines.values()
            )
            text_parts.append(f"{segment_header}\n{segment_text}")

    # Join segments with 3 newline characters
    return "\n\n\n".join(text_parts)
