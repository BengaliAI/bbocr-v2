from bs4 import BeautifulSoup
from bs4.formatter import HTMLFormatter
import math
import numpy as np
from pathlib import Path
import copy 
import os

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
    def __init__(self,filename : str,template_dir : str)-> None:
        """
        This function initializes the class
        Args:
            template_dir (str) : directory for holding HTML templates
            filename (str): Name of the html file to generate
        """
        self.template_dir=template_dir
        # index
        with open(os.path.join(self.template_dir, "index.html"), "r") as f:
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
        with open(os.path.join(self.template_dir, f"{template_name}.html"), "r") as f:
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

    def create_html_file(self, save_path):
        """
        This function creates the html file
        """
        with open(save_path, "w",encoding="utf-8") as f:
            f.write(str(self.index_template.prettify(formatter=HTMLFormatter(indent=2))))
    
    def get_html_data(self):
        """
        returns the raw html data
        """ 
        return str(self.index_template.prettify(formatter=HTMLFormatter(indent=2)))

