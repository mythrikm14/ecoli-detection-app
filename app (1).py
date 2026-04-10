import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.title("🧬 E. coli Detection App")

# -----------------------------
# Google Drive Model Settings
# -----------------------------
FILE_ID = "13UnqcheTgukQEpuNIUs8k6zzwn-RWtY-"
MODEL_NAME = "ecoli_detector.h5"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_NAME)

DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# -----------------------------
# Load Model (Download if needed)
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
st.success("✅ Model loaded successfully")

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload microscope image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prob = model.predict(img)[0][0]

    if prob < 0.5:
        st.error(f"🦠 E. coli detected\nConfidence: {(1 - prob) * 100:.2f}%")
    else:
        st.success(f"✅ Not E. coli\nConfidence: {prob * 100:.2f}%")
