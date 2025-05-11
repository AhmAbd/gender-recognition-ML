import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load trained components
model = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Gender Recognition", layout="centered")

st.title("🧠 Gender Recognition from Face Image")
st.write("Upload a face image (grayscale or color) and the model will predict the gender.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        image_resized = image.resize((32, 32))           # Resize to match training
        img_array = np.array(image_resized).flatten().reshape(1, -1)

        st.image(image_resized, caption="Processed Image", width=150)
        st.write("Image loaded and preprocessed.")

        # Normalize and apply PCA
        img_scaled = scaler.transform(img_array)
        img_pca = pca.transform(img_scaled)

        # Predict
        prediction = model.predict(img_pca)[0]
        proba = model.predict_proba(img_pca)[0]

        label = "👨 Man" if prediction == 0 else "👩 Woman"
        st.success(f"**Prediction:** {label}")
        st.write(f"Confidence: Man {proba[0]*100:.2f}%, Woman {proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {e}")
