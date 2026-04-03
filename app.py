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
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .highlight-top {
        border-left: 5px solid #28a745;
        background-color: #f4fff4;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .scroll-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_value=True)

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

def preprocess_image(image):
    if not HAS_TENSORFLOW:
        return None
    # Resize to 128x128
    img = image.resize((128, 128))
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize (0 to 1)
    img_array = img_array / 255.0
    return img_array

def get_remedy(prediction_class, symptoms):
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
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
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to AI: {e}"

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
        image = Image.open(uploaded_file)
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
                        processed_img = preprocess_image(image)
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
                                <div style="display: flex; justify-content: space-between;">
                                    <strong>{res['class']}</strong>
                                    <span>{res['prob']:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(res['prob'] / 100.0)
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
