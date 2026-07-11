import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Male/Female Eye Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# COLORS
# ----------------------------------------------------
bg_color = "#F5F7FB"
card_color = "#FFFFFF"
sidebar_color = "#E8F0FE"
text_color = "#0F172A"
accent = "#3B82F6"
accent_hover = "#2563EB"
border = "#CBD5E1"
# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------
st.markdown(
f"""
<style>

/* =====================================================
MAIN APP
===================================================== */

.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {{
    background-color: {sidebar_color};
    border-right: 1px solid {border};
}}

section[data-testid="stSidebar"] * {{
    color: {text_color} !important;
}}

/* =====================================================
HEADINGS
===================================================== */

h1,h2,h3,h4,h5,h6 {{
    color: {text_color} !important;
    font-weight:700;
}}

p,label,span {{
    color: {text_color} !important;
}}

/* =====================================================
MARKDOWN
===================================================== */

div[data-testid="stMarkdownContainer"] {{
    color:{text_color} !important;
}}

div[data-testid="stMarkdownContainer"] * {{
    color:{text_color} !important;
}}

/* =====================================================
BUTTONS
===================================================== */

div.stButton > button {{

    width:100%;
    height:50px;

    background:{accent};
    color:white;

    border:none;

    border-radius:12px;

    font-size:17px;

    font-weight:bold;
}}

div.stButton > button:hover {{

    background:{accent_hover};
    color:white;

}}

/* =====================================================
LINK BUTTONS
===================================================== */

div[data-testid="stLinkButton"] button {{

    background:{card_color};

    color:white;

    border:1px solid {border};

}}

div[data-testid="stLinkButton"] button *{{
    color:white !important;
}}

/* =====================================================
FILE UPLOADER
===================================================== */

[data-testid="stFileUploader"] {{

    background:{card_color};

    border:2px dashed {accent};

    border-radius:15px;

}}

[data-testid="stFileUploader"] * {{

    color:white !important;

}}

/* =====================================================
EXPANDERS
===================================================== */

div[data-testid="stExpander"] {{

    background:{card_color};

    border-radius:15px;

    border:1px solid {border};

}}

div[data-testid="stExpander"] * {{

    color:white !important;

}}

/* =====================================================
METRICS
===================================================== */

div[data-testid="stMetric"] {{

    background:{card_color};

    border:1px solid {border};

    border-radius:15px;

    padding:15px;

}}

div[data-testid="stMetric"] * {{

    color:white !important;

}}

/* =====================================================
INFO / SUCCESS / WARNING
===================================================== */

div[data-baseweb="notification"] * {{

    color:white !important;

}}

.stAlert *{{
    color:white !important;
}}

/* =====================================================
PROGRESS BAR
===================================================== */

div[data-testid="stProgressBar"] > div {{
    background:{accent};
}}

/* =====================================================
TEXT INPUT
===================================================== */

input {{
    color:white !important;
}}

textarea {{
    color:white !important;
}}

/* =====================================================
PLOTLY
===================================================== */

.js-plotly-plot {{
    background:{card_color};
    border-radius:12px;
}}

/* =====================================================
FOOTER
===================================================== */

.footer{{
    text-align:center;
    color:#CBD5E1;
    font-size:14px;
}}

</style>
""",
unsafe_allow_html=True
)
