#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import streamlit as st
# intro
st.set_page_config(layout="wide")

import base64
from PIL import Image
import numpy as np
import requests
import pandas as pd
import cv2 
from bbocrv2.ocr import ImageOCR
from bbocrv2.visualize import  draw_word_polys,draw_document_layout
#--------------------------------------------------
# main
#--------------------------------------------------

INFO="""
| Task | Model  | Module   |
|-----------|-----------|-----------|
| Text Detection | Differential Binarizer (Word)| PaddleOCR |
| Text Recognition| ApsisNet (bangla)|ApsisOCR |
| Document Layout Analysis| Yolov8-DLA (DLSprint-BadLad)| BBOCR |
| Reading Order detection | DBScan | BBOCRv2|
"""

@st.cache_resource
def load_model():
    ocr=ImageOCR()
    return ocr

ocr=load_model()

def main():
    
    st.title("bbocrv2: Improved Bangla text word detection,recognition and layout analysis")
    
    with st.sidebar:
        st.markdown("**Module and Model List**")
        st.markdown(INFO)
    
    # For newline
    st.write("\n")
    
    # File selection
    st.title("Document selection")
    # Choose your own image
    uploaded_file = st.file_uploader("Upload files", type=["png", "jpeg", "jpg"])
    
    # For newline
    st.write("\n")
    # Instructions
    st.markdown("*click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1,1,1,1,1))
    cols[0].subheader("Input Image")
    cols[1].subheader("Processed Image")
    cols[2].subheader("Word Detection")
    cols[3].subheader("Document Layout")
    cols[4].subheader("Text and Reading Order")
    
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        arr = np.array(image)
        cols[0].image(arr)
        with st.spinner('Executing OCR'):
            output=ocr(arr)
        
        cols[1].image(output["rotation"]["rotated_image"])
        # word-detection
        word_det_viz=draw_word_polys(output["rotation"]["rotated_image"],[entry["poly"] for entry in output["words"]])
        cols[2].image(word_det_viz)
        # layout 
        layout_viz=draw_document_layout(output["rotation"]["rotated_image"],output["segments"])
        cols[3].image(layout_viz)
        # recognition and rdo
        df=pd.DataFrame(output["words"])
        df=df[['text','line_num','word_num']]
        cols[4].dataframe(df)
        
                
if __name__ == '__main__':  
    main()