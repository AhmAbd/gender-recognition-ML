import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import os
import warnings

# Suppress XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit app title
st.title("Gender Recognition App")
st.write("Choose to either upload an image or use your webcam to predict the gender (Man or Woman).")

# Paths to saved artifacts (artifacts are in the same directory as this script)
ARTIFACTS_DIR = ""  # Empty string means the current directory
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
PCA_PATH = os.path.join(ARTIFACTS_DIR, "pca.pkl")
SELECTOR_PATH = os.path.join(ARTIFACTS_DIR, "selector.pkl")

# Load the saved artifacts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    selector = joblib.load(SELECTOR_PATH)
except FileNotFoundError as e:
    st.error(f"Error: Could not find artifact files. Please ensure model.pkl, scaler.pkl, pca.pkl, and selector.pkl are in the same directory as this script.")
    st.stop()

# Debug: Check the number of features the scaler expects
expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
if expected_features is not None:
    st.write(f"Scaler expects {expected_features} features.")
    if expected_features != 2816:
        st.error(f"Scaler feature mismatch! The scaler expects {expected_features} features, but the pipeline generates 2816 features. Please retrain the model with the updated training script and replace the artifacts.")
        st.stop()
else:
    st.warning("Could not determine the number of features expected by the scaler. Proceeding, but this may cause errors.")

# Load VGG16 for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
vgg_model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

# Face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    st.error("Haar cascade file not found. Please ensure cv2.data.haarcascades is available.")
    st.stop()
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load face cascade classifier.")
    st.stop()

# Function to extract VGG16 features
def extract_vgg_features(image):
    img_array = img_to_array(cv2.resize(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), (48, 48)))
    img_array = np.expand_dims(img_array, axis=0)
    features = vgg_model.predict(img_array, verbose=0)
    return features.reshape(1, -1)

# Preprocess the image (from either webcam or file upload)
def preprocess_image(image):
    # Convert to grayscale
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=8, minSize=(40, 40))
    if len(faces) == 0:
        st.warning("No face detected in the image. Please try again with a different image or capture.")
        return None
    
    # Use the first detected face
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]
    
    # Resize to 48x48
    face_img = Image.fromarray(face_img).resize((48, 48))
    face_array = np.array(face_img).flatten()  # 2304 features
    
    # Extract VGG16 features
    vgg_features = extract_vgg_features(np.array(face_img))  # 512 features
    
    # Combine raw pixels and VGG features
    combined_features = np.concatenate([face_array, vgg_features.flatten()])  # 2304 + 512 = 2816 features
    
    # Verify feature count
    if combined_features.shape[0] != 2816:
        st.error(f"Feature count mismatch: Expected 2816 features, but got {combined_features.shape[0]}.")
        return None
    
    # Scale features
    combined_features = scaler.transform([combined_features])
    
    # Apply PCA if used
    if pca is not None:
        combined_features = pca.transform(combined_features)
    
    # Apply feature selector
    if hasattr(selector, 'transform'):
        combined_features = selector.transform(combined_features)
    
    return combined_features

# Let the user choose input method
input_method = st.radio("Choose input method:", ("Webcam", "Upload Image"))

# Get the image based on the chosen input method
if input_method == "Webcam":
    captured_image = st.camera_input("Take a picture with your webcam")
else:
    captured_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the image if one is provided
if captured_image is not None:
    # Display the captured or uploaded image
    image = Image.open(captured_image)
    st.image(image, caption="Captured/Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    features = preprocess_image(image)
    if features is not None:
        # Predict gender
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
        
        # Display prediction
        gender = "Man" if prediction[0] == 0 else "Woman"
        st.subheader(f"Predicted Gender: {gender}")
        
        if prediction_proba is not None:
            confidence_man = prediction_proba[0] * 100
            confidence_woman = prediction_proba[1] * 100
            st.write(f"Confidence (Man): {confidence_man:.2f}%")
            st.write(f"Confidence (Woman): {confidence_woman:.2f}%")