
#--------------------------------
# imports
#--------------------------------
import streamlit as st
import json
import os
import cv2
import time
import math
import numpy as np
import gdown
from PIL import Image
from apsisocr import ApsisBNOCR
from ultralytics import YOLO
from bbocr.inference import run_inference,clean_region_of_interests
from bbocr.utils import generate_html
#-----------------------------
# models
#-----------------------------

# YOLO
YOLO_DLA_GID="1n-XbOwUwgMjaFPFzEJ59Avrl9Nc5xsx8"  # Google Drive Link of Yolo Model Weights
# local weight file path
yolo_model_weight_path = "weights/best.pt"
# download if not found
if not os.path.isfile(yolo_model_weight_path ):
    download(YOLO_DLA_GID,yolo_model_weight_path )


@st.cache_resource
def load_models():
    dla= YOLO(yolo_model_weight_path)
    ocr=ApsisBNOCR()
    return dla,ocr

dla,ocr=load_models()

template_dir="templates/"
#-----------------------------
# main
#-----------------------------

# # Simulated user database for login/signup
# users = {"admin": "admin123"}

# # # Function to make an API call to your OCR system
# # def call_ocr_api(image_bytes):
# #     api_url = "http://<your-ocr-api-endpoint>"  # Replace with your OCR API URL
# #     files = {"file": image_bytes}
# #     response = requests.post(api_url, files=files)
    
# #     if response.status_code == 200:
# #         # Assuming the API returns plain text or JSON with extracted text
# #         return response.text  # Adjust based on your API's response format
# #     else:
# #         return f"Error: {response.status_code} - {response.text}"

# # Login and Signup Pages
# def login_signup():
#     st.title("Login / Signup")

#     login_option = st.radio("Choose an option", ("Login", "Signup"))

#     if login_option == "Login":
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         if st.button("Login"):
#             if username in users and users[username] == password:
#                 st.session_state["logged_in"] = True
#                 st.success("Login successful!")
#             else:
#                 st.error("Invalid username or password.")
#     elif login_option == "Signup":
#         username = st.text_input("Create a Username")
#         password = st.text_input("Create a Password", type="password")
#         if st.button("Signup"):
#             if username in users:
#                 st.warning("User already exists. Please login.")
#             else:
#                 users[username] = password
#                 st.success("Signup successful! Please login.")

# Main Application Page
def main():
    st.title("BBOCR")
    uploaded_file = st.file_uploader("Upload an Image file",  type=["png", "jpeg", "jpg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
        if st.button("Generate HTML"):
            with st.spinner("Processing..."):
                # image saving
                image = Image.open(uploaded_file)
                arr = np.array(image)
                os.makedirs(os.path.join(os.getcwd(),"images"),exist_ok=True)
                image.save("images/data.png")
                # ocr content
                region_of_interests=run_inference("images/data.png",
                                                  os.path.join(os.getcwd(),"images"),
                                                  dla,
                                                  ocr)
            
            region_of_interests=clean_region_of_interests(region_of_interests)

            # html 
            print("Generated HTML")
            html_content=generate_html(region_of_interests,"images/data.html",template_dir)
            # Display the HTML content
            st.components.v1.html(html_content, height=600, scrolling=True)
            
            # if result_text.startswith("Error"):
            #     st.error(result_text)
            # else:
            #     html_content = f"<html><body><p>{result_text}</p></body></html>"
            #     st.success("HTML generated successfully!")

            #     # Provide a downloadable HTML file
            #     st.download_button(
            #         label="Download HTML",
            #         data=html_content,
            #         file_name="output.html",
            #         mime="text/html"
            #     )

# # Main Logic
# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False

# if st.session_state["logged_in"]:
#     ocr_page()
# else:
#     login_signup()

if __name__=="__main__":
    main()