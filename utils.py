from bs4 import BeautifulSoup
from bs4.formatter import HTMLFormatter
import math
import numpy as np
from pathlib import Path
import copy 

INFO_DICT = {
    "class": None,
    "coordinates": None,
    "left": None,
    "top": None,
    "elem_height": None,
    "elem_width": None,
    "img_height": None,
    "img_width": None,
    "text": None,
    "single-line": False,
    "img_src": None,
}

names = {0: "paragraph", 1: "text_box", 2: "image", 3: "table"}

class HtmlGenerator:
    """
    This class generates the html file
    """
    def __init__(self, filename="default"):
        """
        This function initializes the class
        Args:
        filename: Name of the html file
        """
        with open("reconstruction/templates/index.html", "r") as f:
            index_template = f.read()

        self.index_template = BeautifulSoup(index_template, "html.parser")
        self.index_template_root_div = self.index_template.find("div", {"id": "root"})
        self.filename = filename

    def read_html_template(self, template_name):
        """
        This function reads the html template
        Args:
        template_name: Name of the template
        Returns:
        soup_template: Template
        """
        with open(f"reconstruction/templates/{template_name}.html", "r") as f:
            template = f.read()
            soup_template = BeautifulSoup(template, "html.parser")
            return soup_template

    def get_styles(self, dict):
        """
        This function gets the styles for the html elements
        Args:
        dict: Dictionary containing the styles
        Returns:
        styles: Styles for the html elements
        """
        styles = f'top: {dict["top"]}vh; left: {dict["left"]}vw; height: {dict["elem_height"]}vh; width: {dict["elem_width"]}vw;'
        return styles

    def insert_paragraph(self, paragraph_info):
        """
        This function inserts the paragraph into the html file
        Args:
        paragraph_info: Information about the paragraph
        """
        paragraph_template = self.read_html_template("paragraph")

        p_tag = paragraph_template.find("p")
        text = paragraph_template.new_string(paragraph_info["text"])
        p_tag.append(text)

        paragraph_div = paragraph_template.find("div")
        paragraph_div["style"] = self.get_styles(paragraph_info)

        self.index_template_root_div.append(paragraph_template)

    def insert_text_box(self, text_box_info):
        """
        This function inserts the text box into the html file
        Args:
        text_box_info: Information about the text box
        """
        text_box_template = self.read_html_template("text_box")

        p_tag = text_box_template.find("p")
        text = text_box_template.new_string(text_box_info["text"])
        p_tag.append(text)

        text_box_div = text_box_template.find("div")
        text_box_div["style"] = self.get_styles(text_box_info)

        self.index_template_root_div.append(text_box_template)

    def insert_image(self, img_info):
        """
        This function inserts the image into the html file
        Args:
        img_info: Information about the image
        """
        image_template = self.read_html_template("image")

        img_div = image_template.find("div")
        img_div["style"] = self.get_styles(img_info)

        img_tag = image_template.new_tag("img")
        img_tag["src"] = img_info["img_src"]

        img_style = "width: 100%; height: 100%; object-fit: fill;"
        img_tag["style"] = img_style

        img_div.append(img_tag)

        self.index_template_root_div.append(image_template)

    def create_html_file(self, img_src_save_dir):
        """
        This function creates the html file
        """
        img_src_save_dir
        html_path = Path(img_src_save_dir).parent
        with open(html_path / f"reconstruction/{self.filename}.html", "w") as f:
            f.write(
                str(self.index_template.prettify(formatter=HTMLFormatter(indent=2)))
            )

# Helper function to initialize HTMLGenerator and passing element where needed

def generate_html(detected_elements_info, img_src_save_dir, file_name):
    """
    This function generates the html file
    Args:
    detected_elements_info: Information about the detected elements
    file_name: Name of the file
    """
    file_name, extension = file_name.split(".")

    gen = HtmlGenerator(file_name)

    for element_info in detected_elements_info:

        if element_info["class"] == "paragraph":
            gen.insert_paragraph(element_info)

        elif element_info["class"] == "text_box":
            gen.insert_text_box(element_info)

        elif element_info["class"] == "image":
            gen.insert_image(element_info)

    gen.create_html_file(img_src_save_dir)

# Helper function to get proper coordinate information and padding

def get_normalized_coordinates(xyxy_tensor, height, width):
    """
    This function gets the normalized coordinates
    Args:
    xyxy_tensor: Tensor containing the coordinates
    height: Height of the image
    width: Width of the image
    Returns:
    coordinates: Normalized coordinates
    """
    x_min = xyxy_tensor[0][0].item() / width
    y_min = xyxy_tensor[0][1].item() / height
    x_max = xyxy_tensor[0][2].item() / width
    y_max = xyxy_tensor[0][3].item() / height

    coordinates = [x_min, y_min, x_max, y_max]
    return coordinates


def get_original_coordinates(normalized_coordinates, image_width, image_height):
    """
    This function gets the original coordinates
    Args:
    normalized_coordinates: Normalized coordinates
    image_width: Width of the image
    image_height: Height of the image
    Returns:
    orig_coordinates: Original coordinates
    """
    orig_coordinates = [None] * 4

    orig_coordinates[0] = math.floor(normalized_coordinates[0] * image_width)
    orig_coordinates[1] = math.floor(normalized_coordinates[1] * image_height)
    orig_coordinates[2] = math.ceil(normalized_coordinates[2] * image_width)
    orig_coordinates[3] = math.ceil(normalized_coordinates[3] * image_height)

    return orig_coordinates




def top_bottom_padding(cropped_text_region):
    """
    This function adds top and bottom padding to the cropped text region
    Args:
    cropped_text_region: Cropped text region
    Returns:
    padded_image: Padded image
    """
    h, w = cropped_text_region.shape[:2]
    padded_height = int(h * 1.5)
    padded_width = w

    padded_image = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * 255

    top_padding = (h * 2 - h) // 2
    bottom_padding = top_padding + h

    padded_image[top_padding:bottom_padding, :] = cropped_text_region

    return padded_image


def process_html_block(data_box):
    info_dict = copy.deepcopy(INFO_DICT)
    cls = data_box.cls.item()
    img_height, img_width = data_box.orig_shape
    normalized_coordinates = get_normalized_coordinates(
        data_box.xyxy, img_height, img_width
    )
    if cls == 0:
        info_dict["class"] = names[0]
    elif cls == 1:
        info_dict["class"] = names[1]
    elif cls == 2:
        info_dict["class"] = names[2]
    elif cls == 3:
        info_dict["class"] = names[3]
    info_dict["coordinates"] = normalized_coordinates
    info_dict["left"], info_dict["top"] = (
        normalized_coordinates[0] * 100,
        normalized_coordinates[1] * 100,
    )
    info_dict["img_height"], info_dict["img_width"] = img_height, img_width
    info_dict["elem_width"] = (normalized_coordinates[2] - normalized_coordinates[0]) * 100
    info_dict["elem_height"] = (normalized_coordinates[3] - normalized_coordinates[1]) * 100
    return info_dict, normalized_coordinates


