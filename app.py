import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Male/Female Eye Classifier",
    page_icon="👁️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("male_female_eye_model.keras")

model = load_model()

IMG_SIZE = 128

# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=70
    )

    st.title("Eye Classifier")

    st.markdown("---")

    st.write(
        """
### About

This application predicts whether the uploaded
eye image belongs to a

- 👨 Male
- 👩 Female

using a CNN model.
"""
    )

    st.markdown("---")

    st.subheader("Developer")

    st.link_button(
        "💻 GitHub",
        "https://github.com/Akanksha549/"
    )

    st.link_button(
        "🔗 LinkedIn",
        "https://www.linkedin.com/in/akanksha-mishra-7894912bb"
    )

    st.markdown("---")

    st.success("TensorFlow • Keras • Streamlit")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("👁️ Male & Female Eye Classification")

st.caption(
    "Deep Learning based Gender Classification using Human Eye Images"
)

st.divider()

left, right = st.columns([1,1])

with left:

    st.subheader("📂 Upload Eye Image")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

with right:

    st.info(
        """
### Instructions

✔ Upload a clear eye image

✔ JPG / PNG / JPEG

✔ Image will be resized automatically

✔ Prediction takes less than 1 second
"""
    )
