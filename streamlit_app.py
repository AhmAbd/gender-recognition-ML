import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load model components
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
selector = joblib.load("selector.pkl")

st.set_page_config(page_title="Gender Prediction from Face Image", layout="centered")

st.title("Gender Prediction from Face Image")
st.write("Upload a face image (grayscale or colored), the model will predict gender.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Step 1: Read & preprocess image
        image = Image.open(uploaded_file).convert("L")
        image_resized = image.resize((128, 128))  # Match training size
        img_array = np.array(image_resized).flatten().reshape(1, -1)

        st.image(image_resized, caption="Processed Image", width=150)
        st.write("Image loaded and preprocessed.")

        # Step 2: Apply pipeline: Scale → PCA → SVD → SelectKBest
        img_scaled = scaler.transform(img_array)
        img_pca = pca.transform(img_scaled)

        # Apply same SVD projection as training
        U, S, Vt = np.linalg.svd(pca.transform(scaler.transform([img_array.flatten()])), full_matrices=False)
        img_svd = np.dot(img_pca, Vt.T[:, :500])

        # Select best features
        img_selected = selector.transform(img_svd)

        # Step 3: Predict
        prediction = model.predict(img_selected)[0]
        proba = model.predict_proba(img_selected)[0]

        label = "Man" if prediction == 0 else "Woman"
        st.success(f"Prediction: {label}")
        st.write(f"Confidence → Man: {proba[0]*100:.2f}%, Woman: {proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {e}")