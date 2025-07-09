    import streamlit as st
    import numpy as np
    from PIL import Image
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    import os
    import requests # For downloading the file

    # --- Google Drive Integration (Conceptual - Requires actual implementation) ---
    # This part is highly dependent on your Google Drive sharing settings and authentication method.
    # For a publicly shared file, you might use a direct download link.
    # For private files, you'll need Google Drive API and authentication.

    MODEL_FILE_ID = "1jEowV_m9ojgGuE7CAqPgPKKwUGZixWqg" # Replace with your actual file ID
    MODEL_FILENAME = "Modelenv.v1.h5"
    MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME) # Save in current working directory

    @st.cache_resource # Cache the model loading to avoid re-downloading on every rerun
    def load_model_from_drive():
        st.write("Attempting to download model from Google Drive...")
        try:
            # Example for a publicly shared file (replace with your actual download URL)
            # This URL format is common for direct downloads of publicly shared files.
            # You might need to adjust it based on how you shared the file.
            download_url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
            response = requests.get(download_url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes

            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded successfully!")
            return load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error downloading or loading model from Google Drive: {e}")
            st.info("Please ensure the Google Drive file ID is correct and the file is accessible.")
            return None

    # --- Streamlit App Logic ---
    st.title("Satellite Image Land Cover Classifier")
    st.write("Upload a satellite image to classify its land cover type.")

    # Load the model (this function will handle download if needed)
    model = load_model_from_drive()

    if model is None:
        st.stop() # Stop the app if model couldn't be loaded

    # Define the class names
    class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for the model
        img_resized = img.resize((255, 255)) # Resize to target_size used during training
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Rescale pixel values

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(predictions) * 100

        st.success(f"Prediction: **{predicted_class_name}** with {confidence:.2f}% confidence.")

        st.subheader("Prediction Probabilities:")
        for i, prob in enumerate(predictions[0]):
            st.write(f"- {class_names[i]}: {prob*100:.2f}%")

    
