import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import requests
from io import BytesIO
import gdown  # Special package for Google Drive downloads
import zipfile  # Only needed if your model is zipped
# Configuration - IMPORTANT: Update these with your actual Google Drive link
GOOGLE_DRIVE_LINK = "https://drive.google.com/uc?id=1jEowV_m9ojgGuE7CAqPgPKKwUGZixWqg"  # Replace with your actual file ID
LOCAL_MODEL_FILENAME = "Modelenv.v1.h5"
IS_MODEL_ZIPPED = False  # Change to True if your Google Drive file is zipped

# Function to download and load the model
@st.cache_resource(show_spinner=False)
def load_remote_model():
    # Create a progress bar
    progress_bar = st.progress(0, text="Downloading model... (This may take a few minutes)")
    
    try:
        # Step 1: Download the model file
        progress_bar.progress(10, text="Connecting to Google Drive...")
        
        # Google Drive download using gdown
        gdown.download(
            GOOGLE_DRIVE_LINK,
            LOCAL_MODEL_FILENAME,
            quiet=True
        )
        
        progress_bar.progress(40, text="Model downloaded. Verifying...")
        
        # Check if file exists
        if not os.path.exists(LOCAL_MODEL_FILENAME):
            st.error("Failed to download the model file.")
            st.stop()
            
        # Step 2: Handle zipped model if needed
        model_path = LOCAL_MODEL_FILENAME
        
        if IS_MODEL_ZIPPED:
            progress_bar.progress(50, text="Unzipping model...")
            with zipfile.ZipFile(LOCAL_MODEL_FILENAME, 'r') as zip_ref:
                zip_ref.extractall()
            model_path = "Modelenv.v1.h5"  # Update this if the extracted file has a different name
            progress_bar.progress(70, text="Model unzipped successfully.")
        
        # Step 3: Load the Keras model
        progress_bar.progress(80, text="Loading neural network...")
        model = load_model(model_path)
        progress_bar.progress(100, text="Model loaded successfully!")
        progress_bar.empty()  # Remove the progress bar
        
        return model
        
    except Exception as e:
        progress_bar.progress(100, text="Error occurred!")
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None

# --- MAIN APP UI ---
st.title("🛰️ Satellite Image Classification")
st.subheader("Identify land cover from satellite imagery")
st.write("This AI model classifies images into: Cloudy, Desert, Green Area, or Water")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    - Model: CNN trained on satellite imagery
    - Categories: Cloudy, Desert, Green Area, Water
    - Architecture: EfficientNetB0
    - Accuracy: 92% on validation set
    """)
    
    st.divider()
    st.write("The model file is hosted on Google Drive and downloaded when you first use the app.")

# Load the model (but only if we need it)
if st.session_state.model is None:
    with st.spinner("Initializing app (this only happens once)..."):
        st.session_state.model = load_remote_model()
        st.session_state.model_loaded = True

# File uploader
uploaded_file = st.file_uploader(
    "Upload a satellite image (JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    help="Try to use clear satellite images with visible land features"
)

if uploaded_file is not None and st.session_state.model_loaded:
    # Display and process the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Your Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image..."):
        # Preprocess image for the model
        img_resized = img.resize((255, 255))  # Match model's expected input
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = st.session_state.model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx] * 100
        class_names = ['Cloudy', 'Desert', 'Green Area', 'Water']
        pred_class = class_names[pred_idx]
    
    # Display results
    st.success(f"Prediction: **{pred_class}** (Confidence: {confidence:.1f}%)")
    
    # Show confidence breakdown
    st.subheader("Confidence Breakdown")
    for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
        st.progress(float(prob), f"{name}: {prob*100:.1f}%")

# Add some style
st.markdown("""
<style>
    .stProgress > div > div > div {
        background-color: #1abc9c;
    }
    [data-testid="stFileUploader"] {
        padding: 20px;
        border: 2px dashed #ccc;
        border-radius: 8px;
    }
    [data-testid="stMarkdownContainer"] h1 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)
