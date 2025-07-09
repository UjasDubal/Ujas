import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page configuration
st.set_page_config(
    page_title="🛰️ Satellite Image Land Cover Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .prediction-card {
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Class information
CLASS_INFO = {
    'Cloudy': {
        'description': 'Areas covered by clouds in satellite imagery',
        'color': '#87CEEB',
        'icon': '☁️'
    },
    'Desert': {
        'description': 'Arid or semi-arid land areas with sparse vegetation',
        'color': '#F4A460',
        'icon': '🏜️'
    },
    'Green_Area': {
        'description': 'Vegetated areas including forests, grasslands, and agricultural land',
        'color': '#90EE90',
        'icon': '🌳'
    },
    'Water': {
        'description': 'Water bodies including rivers, lakes, and oceans',
        'color': '#00BFFF',
        'icon': '💧'
    }
}

CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']
CLASS_DISPLAY_NAMES = ['Cloudy', 'Desert', 'Green Area', 'Water']

@st.cache_resource
def load_trained_model():
    """Load the trained model. Replace with your actual model path."""
    try:
        # Replace 'Modelenv.v1.h5' with your actual model path
        model_path = 'Modelenv.v1.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            # Create a mock model for demonstration
            st.warning("⚠️ Model file not found. Using mock predictions for demonstration.")
            return create_mock_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return create_mock_model()

def create_mock_model():
    """Create a mock model for demonstration purposes."""
    class MockModel:
        def predict(self, x):
            # Generate random probabilities for demonstration
            np.random.seed(42)  # For consistent results
            probs = np.random.random((1, 4))
            probs = probs / probs.sum()  # Normalize
            return probs
    
    return MockModel()

def preprocess_image(img):
    """Preprocess the uploaded image for model prediction."""
    # Resize image to model input size (255x255)
    img_resized = img.resize((255, 255))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, img_array):
    """Make prediction on the preprocessed image."""
    predictions = model.predict(img_array)
    return predictions[0]

def create_prediction_chart(predictions, class_names):
    """Create a horizontal bar chart for predictions."""
    df = pd.DataFrame({
        'Class': class_names,
        'Confidence': predictions * 100,
        'Color': [CLASS_INFO[cls]['color'] for cls in CLASS_NAMES]
    })
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Class',
        orientation='h',
        color='Color',
        color_discrete_map={row['Color']: row['Color'] for _, row in df.iterrows()},
        title="Prediction Confidence (%)",
        labels={'Confidence': 'Confidence (%)', 'Class': 'Land Cover Type'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    
    return fig

def create_pie_chart(predictions, class_names):
    """Create a pie chart for predictions."""
    colors = [CLASS_INFO[cls]['color'] for cls in CLASS_NAMES]
    
    fig = go.Figure(data=[go.Pie(
        labels=class_names,
        values=predictions * 100,
        hole=0.3,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="Prediction Distribution",
        title_font_size=16,
        height=400
    )
    
    return fig

def main():
    # Header
    st.title("🛰️ Satellite Image Land Cover Classifier")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About This App")
        st.markdown("""
        This application uses deep learning to classify satellite images into four categories:
        - **Cloudy**: Cloud-covered areas
        - **Desert**: Arid/semi-arid regions  
        - **Green Area**: Vegetated regions
        - **Water**: Water bodies
        """)
        
        st.header("🚀 How to Use")
        st.markdown("""
        1. Upload a satellite image
        2. Click 'Classify Image'
        3. View the prediction results
        4. Explore confidence scores
        """)
        
        st.header("ℹ️ Model Info")
        st.info("CNN model trained on satellite imagery with 4 land cover classes")
        
        # Class information
        st.header("🏷️ Class Information")
        for class_name, info in CLASS_INFO.items():
            with st.expander(f"{info['icon']} {class_name.replace('_', ' ')}"):
                st.write(info['description'])
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a satellite image to classify its land cover type"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image_pil.size}")
            st.write(f"**Image Mode:** {image_pil.mode}")
            st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
            
            # Classify button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("🔄 Analyzing image..."):
                    try:
                        # Load model
                        model = load_trained_model()
                        
                        # Preprocess image
                        img_array = preprocess_image(image_pil)
                        
                        # Make prediction
                        predictions = predict_image(model, img_array)
                        
                        # Store results in session state
                        st.session_state.predictions = predictions
                        st.session_state.predicted_class = CLASS_NAMES[np.argmax(predictions)]
                        st.session_state.confidence = np.max(predictions)
                        
                        st.success("✅ Classification complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")
    
    with col2:
        st.header("📊 Results")
        
        if hasattr(st.session_state, 'predictions'):
            predictions = st.session_state.predictions
            predicted_class = st.session_state.predicted_class
            confidence = st.session_state.confidence
            
            # Main prediction result
            class_info = CLASS_INFO[predicted_class]
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="color: {class_info['color']};">
                    {class_info['icon']} {predicted_class.replace('_', ' ')}
                </h2>
                <h3>Confidence: {confidence:.1%}</h3>
                <p>{class_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.subheader("🎯 Detailed Confidence Scores")
            
            for i, (class_name, display_name) in enumerate(zip(CLASS_NAMES, CLASS_DISPLAY_NAMES)):
                conf_score = predictions[i]
                class_info = CLASS_INFO[class_name]
                
                col_icon, col_name, col_score = st.columns([1, 3, 2])
                
                with col_icon:
                    st.write(class_info['icon'])
                with col_name:
                    st.write(f"**{display_name}**")
                with col_score:
                    st.write(f"{conf_score:.1%}")
                
                # Progress bar
                st.progress(conf_score)
                st.write("")
            
            # Charts
            st.subheader("📈 Visualization")
            
            # Tabs for different chart types
            tab1, tab2 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
            
            with tab1:
                fig_bar = create_prediction_chart(predictions, CLASS_DISPLAY_NAMES)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab2:
                fig_pie = create_pie_chart(predictions, CLASS_DISPLAY_NAMES)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            
            results_df = pd.DataFrame({
                'Class': CLASS_DISPLAY_NAMES,
                'Confidence': predictions,
                'Confidence (%)': predictions * 100
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv"
            )
            
        else:
            st.info("👆 Upload an image and click 'Classify Image' to see results here.")
            
            # Show sample predictions for demonstration
            st.subheader("📋 Sample Output")
            st.write("This is what the results will look like:")
            
            # Mock sample data
            sample_predictions = np.array([0.15, 0.05, 0.75, 0.05])
            sample_class = 'Green_Area'
            sample_confidence = 0.75
            
            class_info = CLASS_INFO[sample_class]
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: {class_info['color']};">
                    {class_info['icon']} {sample_class.replace('_', ' ')}
                </h3>
                <p>Confidence: {sample_confidence:.1%}</p>
                <small><i>Sample prediction</i></small>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🛰️ Satellite Image Land Cover Classifier | Built with Streamlit & TensorFlow</p>
        <p><small>Upload high-quality satellite images for best results</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()