import streamlit as st
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from PIL import Image
import numpy as np
import google.generativeai as genai
from fpdf import FPDF
import os
import random

# --- Configuration ---
st.set_page_config(
    page_title="Vitamin Deficiency Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini API Configuration
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyD-wj0efwhCkOnaGlyJ7XYxFhK4g4RHm74")
genai.configure(api_key=GEMINI_API_KEY)

# Model Path
MODEL_PATH = "vitamin_model.keras" # Expected in the same directory

# Classes
CLASS_NAMES = ["Vitamin B2 Deficiency", "Vitamin B3 Deficiency", "Vitamin C Deficiency"]

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .prediction-card {
        background-color: #ffffff;
        color: #212529 !important; /* Force dark text color */
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border: 1px solid #dee2e6;
    }
    .prediction-card strong {
        color: #212529 !important;
    }
    .highlight-top {
        border-left: 5px solid #28a745;
        background-color: #f8fff9;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        background-color: #f8fff9;
        padding: 5px;
        border-radius: 5px;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        background-color: #fff5f5;
        padding: 5px;
        border-radius: 5px;
    }
    .scroll-container {
        max-height: 450px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 10px;
        background: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Functions ---

@st.cache_resource
def load_vitamin_model():
    if not HAS_TENSORFLOW:
        st.info("💡 Running in **Simulation Mode** (TensorFlow not installed locally).")
        return "SIMULATION_MODE"
        
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return "SIMULATION_MODE"
    else:
        st.warning(f"Model file '{MODEL_PATH}' not found. Using **Simulation Mode**.")
        return "SIMULATION_MODE"

def preprocess_image(image, target_size=(224, 224)):
    if not HAS_TENSORFLOW:
        return None
    # Resize to target size (detected from model)
    img = image.resize(target_size)
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize (0 to 1)
    img_array = img_array / 255.0
    return img_array

def get_remedy(prediction_class, symptoms):
    # Prompt Template
    prompt = f"""
    A user has been predicted to have {prediction_class} based on a skin image analysis.
    The user also reports the following symptoms: {symptoms}

    Please provide a structured response with the following sections (use emojis and markdown):
    1. 🧬 Cause: Explain the common causes of this deficiency.
    2. 🥦 Remedies (Diet + Lifestyle): Suggest specific foods and lifestyle changes.
    3. ⚠️ Precautions: What should the user avoid?
    4. 🩺 When to consult a Doctor: Specific warning signs.

    Note: This is an AI-generated suggestion, not medical advice.
    """
    
    try:
        # First, try to list models and find one that supports 'generateContent'
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not available_models:
            return "No generative models found for this API key. Please check your Google AI Studio account."
            
        # Prioritize 1.5 models if available
        preferred_order = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        model_to_use = available_models[0] # Default to first one found
        
        for p in preferred_order:
            if p in available_models:
                model_to_use = p
                break
                
        model = genai.GenerativeModel(model_to_use)
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Final Connection Error: {e}. Please ensure your API Key is correct and your region supports Gemini AI."

def generate_pdf(prediction_class, confidence, symptoms, remedy_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Vitamin Deficiency Detector Report", ln=True, align='C')
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Top Prediction: {prediction_class}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Symptoms Reported:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=symptoms)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="AI Recommended Remedies & Advice:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt=remedy_text.encode('latin-1', 'replace').decode('latin-1'))
    
    return pdf.output(dest='S').encode('latin-1')

# --- UI Layout ---

st.title("🩺 Vitamin Deficiency Detector")
st.markdown("### Multimodal AI Analysis for Skin Health")

# Sidebar
with st.sidebar:
    st.header("Settings & Info")
    st.info("Upload a clear image of the affected skin area and describe your symptoms for a comprehensive analysis.")
    st.markdown("---")
    st.write("Built with Streamlit, TensorFlow, and Gemini AI.")

# Main Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload & Input")
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    symptoms = st.text_area("✍️ Enter your symptoms (e.g., fatigue, mouth sores, dry skin):", placeholder="Describe how you feel...")

    analyze_button = st.button("🔍 Analyze Deficiency")

with col2:
    st.subheader("📊 Analysis Results")
    
    if analyze_button:
        if uploaded_file is None:
            st.error("Please upload an image first!")
        elif not symptoms.strip():
            st.warning("Please enter some symptoms for better context.")
        else:
            model = load_vitamin_model()
            
            if model:
                with st.spinner("🔄 Analyzing image and symptoms..."):
                    if model == "SIMULATION_MODE":
                        # Simulate predictions for testing UI without TensorFlow
                        probs = [random.uniform(0.1, 0.9) for _ in range(3)]
                        # Normalize to sum to 1
                        total = sum(probs)
                        predictions = [p/total for p in probs]
                    else:
                        # Real Prediction logic
                        # Detect expected input size (e.g., 224, 224)
                        try:
                            input_shape = model.input_shape
                            # input_shape is usually (None, Height, Width, Channels)
                            target_h, target_w = input_shape[1], input_shape[2]
                        except:
                            target_h, target_w = 224, 224 # Default fallback
                            
                        processed_img = preprocess_image(image, target_size=(target_w, target_h))
                        predictions = model.predict(processed_img)[0]
                    
                    # Sort results
                    results = []
                    for i, prob in enumerate(predictions):
                        results.append({"class": CLASS_NAMES[i], "prob": prob * 100})
                    
                    results = sorted(results, key=lambda x: x['prob'], reverse=True)
                    top_prediction = results[0]
                    
                    # 2. Display probabilities
                    st.write("#### Prediction Probabilities")
                    with st.container():
                        st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
                        for res in results:
                            is_top = (res == top_prediction)
                            card_class = "prediction-card highlight-top" if is_top else "prediction-card"
                            
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <strong>{res['class']}</strong>
                                    <span style="color: #212529;">{res['prob']:.1f}%</span>
                                </div>
                                <div style="width: 100%; background-color: #e9ecef; border-radius: 5px; height: 10px;">
                                    <div style="width: {res['prob']}%; background-color: {'#28a745' if is_top else '#007bff'}; height: 10px; border-radius: 5px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 3. Confidence Logic
                    if top_prediction['prob'] > 80:
                        st.markdown(f'<p class="confidence-high">✅ High Confidence Prediction: {top_prediction["class"]}</p>', unsafe_allow_html=True)
                    elif top_prediction['prob'] < 60:
                        st.markdown(f'<p class="confidence-low">⚠️ Warning: Low Confidence. Please consult a professional.</p>', unsafe_allow_html=True)
                    
                    # 4. Gemini AI Remedies
                    st.markdown("---")
                    st.subheader("💡 AI Recommended Remedies")
                    remedy_text = get_remedy(top_prediction['class'], symptoms)
                    st.markdown(remedy_text)
                    
                    # 5. PDF Generation
                    pdf_bytes = generate_pdf(top_prediction['class'], top_prediction['prob'], symptoms, remedy_text)
                    st.download_button(
                        label="📄 Download Full Report (PDF)",
                        data=pdf_bytes,
                        file_name="Deficiency_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("Problem with model loading. Check logs.")

# Footer
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and does not substitute professional medical advice. Always consult a healthcare provider.")
