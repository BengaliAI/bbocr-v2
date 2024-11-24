from typing import List, Dict, Any


def generate_html(segments_data: List[Dict[str, Any]], page_width: int, page_height: int) -> str:
    """
    Generate an HTML representation of segments data, maintaining relative positions and alignments.

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
        page_width (int): The width of the page or canvas for scaling purposes.
        page_height (int): The height of the page or canvas for scaling purposes.

    Returns:
        str: An HTML string representing the segments with relative positioning and alignment.
    """
    def calculate_bbox_from_poly(poly: List[int]) -> List[int]:
        """Convert a polygon into a bounding box [x_min, y_min, x_max, y_max]."""
        xs = poly[0::2]
        ys = poly[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]

    # Start HTML structure
    html = [
        f"<div style='position:relative; width:{page_width}px; height:{page_height}px; border:1px solid black;'>"
    ]

    for segment in segments_data:
        if segment["label"] in ["paragraph", "text_box"] and segment["data"]:
            # Extract bounding box for the segment
            seg_bbox = segment["bbox"]
            seg_style = (
                f"position:absolute; left:{seg_bbox[0]}px; top:{seg_bbox[1]}px; "
                f"width:{seg_bbox[2] - seg_bbox[0]}px; height:{seg_bbox[3] - seg_bbox[1]}px; overflow:hidden;"
            )
            html.append(f"<div style='{seg_style}'>")

            # Organize words into lines based on relative_line_num
            lines = {}
            for word in segment["data"]:
                line_num = word["relative_line_num"]
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(word)

            # Sort lines and words
            for line_num in sorted(lines.keys()):
                line_words = sorted(lines[line_num], key=lambda w: w["relative_word_num"])
                html.append("<div style='position:relative; display:block; white-space:nowrap;'>")

                for word in line_words:
                    word_bbox = calculate_bbox_from_poly(word["poly"])
                    word_width = word_bbox[2] - word_bbox[0]
                    word_height = word_bbox[3] - word_bbox[1]
                    word_style = (
                        f"position:relative; display:inline-block; "
                        f"width:{word_width}px; height:{word_height}px; "
                        f"vertical-align:top; margin:0; padding:0;"
                    )
                    html.append(f"<span style='{word_style}'>{word['text']}</span>")

                html.append("</div>")  # Close line div

            html.append("</div>")  # Close segment div

    html.append("</div>")  # Close page div

    return "\n".join(html)
