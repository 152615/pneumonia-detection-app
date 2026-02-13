import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Google Drive file ID
file_id = "133h_f_8KVWGe52Um14PMhvn2PtSbi7ny"
url = "https://drive.google.com/uc?id=" + file_id

# Download model
gdown.download(url, "pneumonia_model.keras", quiet=False)


# Load model
model = load_model("pneumonia_model.keras")

st.title("🫁 Pneumonia Detection App")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Prediction: Pneumonia")
    else:
        st.success("✅ Prediction: Normal")



