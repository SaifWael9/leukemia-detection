import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os 

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Leukemia AI Diagnostics", page_icon="🩸", layout="wide")

# ---------------------------------------------------------
# 2. LOAD MODELS (Cached so they only load once)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_resnet_model():
    model_path = os.path.join(BASE_DIR, 'Resnet50.keras')
    return tf.keras.models.load_model(model_path, compile=False) 

@st.cache_resource
def load_effnet_model():
    model_path = os.path.join(BASE_DIR, 'efficientnetv2s.keras')
    return tf.keras.models.load_model(model_path, compile=False)

resnet_model = load_resnet_model()
effnet_model = load_effnet_model()

# ---------------------------------------------------------
# 3. PREPROCESSING FUNCTION
# ---------------------------------------------------------
def preprocess_image(image, model_type):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == 'resnet':
        return tf.keras.applications.resnet50.preprocess_input(img_array)
    elif model_type == 'efficientnet':
        return tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

# ---------------------------------------------------------
# 4. SIDEBAR & UI LAYOUT
# ---------------------------------------------------------
st.sidebar.title("⚙️ AI Configuration")
st.sidebar.write("Select the diagnostic engine:")
model_choice = st.sidebar.radio(
    "Available Models:",
    ("ResNet50 Elite (Primary)", "EfficientNetV2-S (Secondary)", "Compare Both Models")
)

st.title("🩸 Automated Leukemia Detection System")
st.markdown("Upload a pre-segmented mononuclear blood cell image for AI analysis.")

# File Uploader
uploaded_file = st.file_uploader("Upload Blood Smear (.jpg, .png, .jpeg, .bmp)", type=["jpg", "png", "jpeg", "bmp"])

# ---------------------------------------------------------
# 5. PREDICTION LOGIC
# ---------------------------------------------------------
if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        image = Image.open(uploaded_file)
        # FIXED STREAMLIT WARNING: Changed to use_container_width
        st.image(image, caption="Uploaded Mononuclear Cell", use_container_width=True)

    st.markdown("---")
    
    if st.button("🔍 Run AI Analysis", use_container_width=True):
        st.spinner("Analyzing cellular morphology...")
        
        # Scenario A: ResNet50 Only
        if model_choice == "ResNet50 Elite (Primary)":
            processed_img = preprocess_image(image, 'resnet')
            prediction = resnet_model.predict(processed_img)[0]
            
            # FIXED LOGIC: Index 0 is ALL (Leukemia), Index 1 is HEM (Healthy)
            prob_leukemia = prediction[0]
            prob_healthy = prediction[1]
            
            st.subheader("ResNet50 Elite Diagnosis")
            if prob_leukemia > 0.5:
                st.error(f"🚨 ALERT: Leukemic Blast Detected! (Confidence: {prob_leukemia*100:.2f}%)")
            else:
                st.success(f"✅ Normal Cell (Confidence: {prob_healthy*100:.2f}%)")
                
        # Scenario B: EfficientNet Only
        elif model_choice == "EfficientNetV2-S (Secondary)":
            processed_img = preprocess_image(image, 'efficientnet')
            prediction = effnet_model.predict(processed_img)[0]
            
            prob_leukemia = prediction[0]
            prob_healthy = prediction[1]
            
            st.subheader("EfficientNetV2-S Diagnosis")
            if prob_leukemia > 0.5:
                st.error(f"🚨 ALERT: Leukemic Blast Detected! (Confidence: {prob_leukemia*100:.2f}%)")
            else:
                st.success(f"✅ Normal Cell (Confidence: {prob_healthy*100:.2f}%)")
                
        # Scenario C: Compare Both
        elif model_choice == "Compare Both Models":
            st.subheader("⚖️ Comparative Model Analysis")
            
            colA, colB = st.columns(2)
            
            # Run ResNet
            with colA:
                st.markdown("### ResNet50 (Primary)")
                res_processed = preprocess_image(image, 'resnet')
                res_pred = resnet_model.predict(res_processed)[0]
                
                res_prob_leukemia = res_pred[0]
                res_prob_healthy = res_pred[1]
                
                if res_prob_leukemia > 0.5:
                    st.error(f"Prediction: **Leukemia**\n\nConfidence: {res_prob_leukemia*100:.1f}%")
                else:
                    st.success(f"Prediction: **Healthy**\n\nConfidence: {res_prob_healthy*100:.1f}%")
            
            # Run EfficientNet
            with colB:
                st.markdown("### EfficientNetV2-S")
                eff_processed = preprocess_image(image, 'efficientnet')
                eff_pred = effnet_model.predict(eff_processed)[0]
                
                eff_prob_leukemia = eff_pred[0]
                eff_prob_healthy = eff_pred[1]
                
                if eff_prob_leukemia > 0.5:
                    st.error(f"Prediction: **Leukemia**\n\nConfidence: {eff_prob_leukemia*100:.1f}%")
                else:
                    st.success(f"Prediction: **Healthy**\n\nConfidence: {eff_prob_healthy*100:.1f}%")
            
            # Agreement Check
            st.markdown("---")
            res_is_cancer = res_prob_leukemia > 0.5
            eff_is_cancer = eff_prob_leukemia > 0.5
            
            if res_is_cancer == eff_is_cancer:
                st.info("✅ **Diagnostic Consensus:** Both AI models agree on the classification.")
            else:
                st.warning("⚠️ **Diagnostic Discrepancy:** The models disagree. This is a borderline cell requiring manual hematologist review. (Note: ResNet50's cautious bias may flag anomalies that EfficientNet ignores).")