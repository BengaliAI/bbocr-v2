from sklearn.cluster import DBSCAN
import numpy as np
from typing import List, Dict


def assign_line_word_numbers(words: List[Dict[str, List]]) -> List[Dict]:
    """
    Assign reversed line and word numbers to OCR recognition results using DBSCAN clustering.

    Args:
        words (List[Dict]): A list of dictionaries where each dictionary contains:
            - 'poly': [x1, y1, x2, y2, x3, y3, x4, y4] coordinates of the word bounding box
            - 'text': The recognized word text

    Returns:
        List[Dict]: A list of dictionaries with added fields:
            - 'poly': The original bounding box coordinates
            - 'text': The recognized word text
            - 'line_num': The reversed assigned line number (starting from 1)
            - 'word_num': The assigned word number in the line (starting from 1)
    """
    # Calculate the average height of the bounding boxes
    heights = [
        abs(word['poly'][5] - word['poly'][1])  # Height = difference between bottom-left (y4) and top-left (y1)
        for word in words
    ]
    avg_height = np.mean(heights)

    # Calculate the midpoint (x, y) for each word
    midpoints = [
        [(word['poly'][0] + word['poly'][4]) / 2, (word['poly'][1] + word['poly'][5]) / 2]  # Midpoint of top-left and bottom-right
        for word in words
    ]
    midpoints = np.array(midpoints)

    # Apply DBSCAN clustering based on y-coordinate, with eps = avg_height//2
    dbscan = DBSCAN(eps=avg_height//2, min_samples=1)
    line_labels = dbscan.fit_predict(midpoints[:, 1].reshape(-1, 1))  # Use Y-coordinates only

    # Group words by their assigned line labels
    labeled_words = []
    for idx, label in enumerate(line_labels):
        labeled_words.append({**words[idx], 'line_label': label, 'x_mid': midpoints[idx][0]})

    # Sort by line label (reversed Y-coordinate order) and X-coordinates within each line
    labeled_words.sort(key=lambda w: (-w['line_label'], w['x_mid']))

    # Assign reversed line numbers and word numbers
    output = []
    current_line = -1  # Reverse line labels
    current_word = 0
    reversed_line_map = {label: i + 1 for i, label in enumerate(sorted(set(line_labels), reverse=True))}

    for word in labeled_words:
        if word['line_label'] != current_line:
            current_line = word['line_label']
            current_word = 1  # Reset word count for new line
        else:
            current_word += 1

        output.append({
            'poly': word['poly'],
            'text': word['text'],
            'line_num': reversed_line_map[word['line_label']],  # Reversed line number
            'word_num': current_word
        })

    return output
