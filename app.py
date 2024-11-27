import streamlit as st
import base64

# Set the page layout to wide
st.set_page_config(layout="wide")

# Define page titles and corresponding modules
PAGES = {
    "BBOCRv2": "pages/bbocrv2.py",
    "BBOCR": "pages/bbocr.py",
}

def get_data_url(img_path):
    file_ = open(img_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url


# Markdown with icons
flowchart= """
### OCR & Document Layout Analysis System Flow

The following sequence represents the flow of the OCR and Document Layout Analysis system in a circular process.

| **Step**             | **Description** |
|----------------------|-----------------|
| 🧑‍💻 **User**        | Provides the input image (file path or numpy array) |
| 💻 **System**        | Reads and converts the image to RGB |
| 📝 **PaddleDBNet**   | Detects word regions in the image |
| 🔄 **RotationCorrection** | Applies automated rotation correction to the image |
| 🔢 **DBScan**        | Applies reading order detection for words and text |
| 🧠 **APSISNet**      | Performs text recognition on correctly rotated word boxes |
| 📐 **YOLOv8**        | Performs document layout segmentation on the image |
| 🔗 **Merging**       | Merges document segments using vectorized IoU calculation |
| 🗂️ **LayoutHandling** | Checks intersection of merged segments with detected words |
| 📄 **HTMLReconstruction** | Generates the final HTML layout for the image |
| 🔄 **System**        | Returns the final output (HTML with words and document layout) to the user |

"""
# Create a single centered column
col1, col2, col3 = st.columns([1, 10, 1])  # 6 is the width for the center column

with col2:
    # Title of the page
    st.title("bbocrv2: Improved Bangla Text Word Detection, Recognition, and Layout Analysis with HTML Reconstruction")
    # Display the Mermaid flowchart diagram
    st.markdown(flowchart, unsafe_allow_html=True)

    # Intro section
    st.markdown("""
    ---
    # Team Members

    | Name                     | Department          | Registration Number |
    |--------------------------|---------------------|---------------------|
    | **Shattik Bandyopadhyaa** | Software Engineering| 2019831039          |
    | **Anabil Debnath**        | Software Engineering| 2019831071          |

    ---
    """)
    # Info section with table
    st.markdown("# **Module and Model List**")
    st.markdown("""
    | Task | Model  | Module   |
    |-----------|-----------|-----------|
    | Text Detection | Differential Binarizer (Word)| PaddleOCR |
    | Text Recognition| ApsisNet (Bangla)|ApsisOCR |
    | Document Layout Analysis| Yolov8-DLA (DLSprint-BadLad)| BBOCR |
    | Reading Order detection | DBScan | BBOCRv2|
    | HTML Reconstruction | Custom | BBOCRv2|
    """)


    st.markdown("---")
    st.markdown("## **Industry Partner**")
    st.markdown(f'<img src="data:image/gif;base64,{get_data_url("resources/apsis.png")}" alt="apsis">'+'   [apsis solutions limited](https://apsissolutions.com/)',unsafe_allow_html=True)
    st.markdown("## **Research Collaboration**")
    st.markdown(f'<img src="data:image/gif;base64,{get_data_url("resources/bengaliai.png")}" alt="apsis">'+'   [bengali.ai](https://bengali.ai/)',unsafe_allow_html=True)
    st.markdown("---")

