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

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("🖼 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("🤖 Prediction")

        if st.button("🚀 Predict Gender", use_container_width=True):

            with st.spinner("Analyzing image..."):

                start = time.time()

                img = preprocess_image(image)

                prediction = float(
                    model.predict(img, verbose=0)[0][0]
                )

                end = time.time()

                inference_time = end - start

            # -----------------------------------------
            # CLASS MAPPING
            # femaleeyes = 0
            # maleeyes = 1
            # -----------------------------------------

            if prediction >= 0.5:

                label = "👨 Male"

                confidence = prediction

            else:

                label = "👩 Female"

                confidence = 1 - prediction

            st.success(f"### Prediction: {label}")

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            metric2.metric(
                "Inference",
                f"{inference_time:.3f} sec"
            )

            st.progress(confidence)

            st.divider()

            st.subheader("📊 Prediction Probability")

            female_prob = 1 - prediction
            male_prob = prediction

            fig = go.Figure()

            fig.add_trace(

                go.Bar(

                    x=["Female","Male"],

                    y=[female_prob,male_prob],

                    text=[
                        f"{female_prob*100:.1f}%",
                        f"{male_prob*100:.1f}%"
                    ],

                    textposition="outside",

                    marker_color=[
                        "#EC4899",
                        "#2563EB"
                    ]

                )

            )

            fig.update_layout(

                template="plotly_white",

                height=420,

                yaxis=dict(
                    range=[0,1],
                    title="Probability"
                ),

                xaxis_title="Predicted Class",

                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=20
                ),

                showlegend=False

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            st.subheader("📋 Prediction Summary")

            summary = {
                "Predicted Gender": label,
                "Confidence": f"{confidence*100:.2f}%",
                "Inference Time": f"{inference_time:.3f} sec",
                "Model": "CNN",
                "Input Size": "128 × 128 × 3"
            }

            st.json(summary)
