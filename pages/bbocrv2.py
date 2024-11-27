#-*- coding: utf-8 -*-
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import streamlit as st


import base64
from PIL import Image
import numpy as np
import requests
import pandas as pd
import cv2 
from bbocrv2.ocr import ImageOCR
from bbocrv2.visualize import  draw_word_polys,draw_document_layout
from bbocrv2.postprocessing import process_segments_and_words,construct_text_from_segments
from bbocrv2.htmlgen import generate_html
from apsisocr.utils import correctPadding
#--------------------------------------------------
# main
#--------------------------------------------------


@st.cache_resource
def load_model():
    ocr=ImageOCR()
    return ocr

ocr=load_model()

def main():
    
    st.title("bbocrv2: Improved Bangla text word detection,recognition and layout analysis")
    
    
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
        image = Image.open(uploaded_file).convert("RGB")
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
        # text construction
        st.title("Layout wise text construction")
        segments=output["segments"]
        words=output["words"]
        segmented_data=process_segments_and_words(segments,words)
        layout_text_data=construct_text_from_segments(segmented_data)
        st.text_area("layout text", value=layout_text_data,height=400)
        # Anabil---> Code frem here
        st.title("HTML Recontruction")
        height,width=arr.shape[:2]
        html_data=generate_html(segmented_data,height,width,image)
        st.components.v1.html(html_data, height=600, scrolling=True)
        # Word Analysis
        st.title("Word Analysis")
        crops=ocr.detector.get_crops(output["rotation"]["rotated_image"],[entry["poly"] for entry in output["words"]])
        crops=[correctPadding(crop,(128,1024)) for crop in crops]
        crops=[ crop[:,:pad_w] for (crop,pad_w) in crops]
        data=[{"image": crop,"text":text} for crop,text in zip(crops,[entry["text"] for entry in output["words"]])]
        
        # Custom CSS to center the table
        st.markdown(
            """
            <style>
            .centered-table {
                display: flex;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display the table in the center
        st.markdown('<div class="centered-table">', unsafe_allow_html=True)

        # Iterate over the data in chunks of 5 to create rows
        for i in range(0, len(data), 10):
            cols = st.columns(10)  # Create 5 columns for each row
            
            for j in range(10):
                if i + j < len(data):  # Ensure we don't go out of bounds
                    with cols[j]:  # Access the j-th column in the current row
                        st.image(data[i + j]["image"], caption=data[i + j]["text"], use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        
                
if __name__ == '__main__':  
    main()