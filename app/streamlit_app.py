
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.inference import ASLPredictor
from src.config import CLASS_NAMES


st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model (cached)"""
    return ASLPredictor()


def get_confidence_color(confidence):
    """Get color class based on confidence"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Alphabet Recognition System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
    Real-time American Sign Language alphabet recognition using Deep Learning
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This application uses a deep learning model trained on the ASL Alphabet dataset
        to recognize American Sign Language alphabet signs in real-time.
        """)
        
        st.header("📊 Model Information")
        st.write(f"**Architecture:** MobileNetV2 (Transfer Learning)")
        st.write(f"**Classes:** {len(CLASS_NAMES)}")
        st.write(f"**Input Size:** 224×224 pixels")
        
        st.header("🎯 Supported Classes")
        st.write(", ".join(CLASS_NAMES))
        
        st.header("💡 Tips")
        st.info("""
        - Use clear, well-lit images
        - Center your hand in the frame
        - Use a plain background
        - Ensure fingers are clearly visible
        """)
    
    # Load model
    try:
        predictor = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure the model is trained. Run: `python src/train.py`")
        return
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "📹 Live Webcam", "📚 Reference Chart"])
    
    with tab1:
        st.header("Upload an Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of an ASL alphabet sign"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Predictions")
                
                with st.spinner("Analyzing image..."):
                    results = predictor.predict(image, top_k=5)
                
                # Top prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style='text-align: center; margin: 0;'>
                        Predicted Sign: <span style='color: #1f77b4;'>{results['top_prediction']}</span>
                    </h2>
                    <h3 style='text-align: center; margin-top: 10px;'>
                        Confidence: <span class='{get_confidence_color(results['top_confidence'])}'>
                        {results['top_confidence']:.2%}
                        </span>
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 5 predictions
                st.subheader("Top 5 Predictions")
                for i, pred in enumerate(results['predictions'], 1):
                    confidence = pred['confidence']
                    progress_color = get_confidence_color(confidence)
                    
                    st.write(f"**{i}. {pred['class']}**")
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2%}")
                    st.markdown("---")
    
    with tab2:
        st.header("Live Webcam Recognition")
        st.info("""
        For real-time webcam recognition, please use the standalone application:
        
        ```bash
        python app/webcam_demo.py
        ```
        
        This will open a window with live ASL recognition from your webcam.
        """)
        
        st.markdown("""
        ### How to use:
        1. Run the webcam demo script
        2. Position your hand in the green box
        3. Make an ASL alphabet sign
        4. See real-time predictions!
        5. Press 'q' to quit
        """)
    
    with tab3:
        st.header("ASL Alphabet Reference")
        st.write("Learn the ASL alphabet signs:")
        
        st.info("""
        The model recognizes 29 classes:
        - **A-Z**: Standard alphabet letters
        - **SPACE**: Open palm gesture
        - **DELETE**: Specific gesture for deletion
        - **NOTHING**: No sign (background)
        """)
        
        st.markdown("""
        ### Tips for accurate recognition:
        - **Lighting**: Ensure good, even lighting on your hand
        - **Background**: Use a plain, contrasting background
        - **Hand Position**: Keep your hand centered and steady
        - **Clarity**: Make clear, distinct signs
        - **Distance**: Keep appropriate distance from camera
        """)
        
        st.success("""
        For a comprehensive ASL alphabet chart, visit:
        https://www.startasl.com/asl-alphabet
        """)
    
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #666;'>
    Built with ❤️ using TensorFlow, Keras, and Streamlit | 
    <a href='https://github.com/yourusername/asl-recognition'>GitHub</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
