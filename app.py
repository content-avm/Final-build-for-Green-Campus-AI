import streamlit as st
import tensorflow as tf
import tf_keras as keras  # Crucial fix for DepthwiseConv2D error
from PIL import Image, ImageOps
import numpy as np
import google.generativeai as genai

# --- 1. INTELLIGENCE LAYER (Gemini Setup) ---
# Replace with your actual API Key from Google AI Studio
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. PERCEPTION LAYER (Model Loading) ---
@st.cache_resource
def load_my_model():
    # Use tf_keras to bypass Keras 3 compatibility issues
    model = keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_my_model()

# --- 3. INTERFACE LAYER (Streamlit UI) ---
st.set_page_config(page_title="Green Campus AI", page_icon="🌱")
st.title("♻️ Waste Mismanagement AI Sorter")
st.markdown("### Identify waste and get environmental facts instantly!")

# Camera input for the user
img_file = st.camera_input("Position the waste item in front of the camera")

if img_file:
    # --- Image Pre-processing ---
    image = Image.open(img_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    # Normalization required by Teachable Machine models
    normalized_image_array = (img_array / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # --- Perception: Classify Waste ---
    prediction = model.predict(data)
    index = np.argmax(prediction)
    label = class_names[index]
    confidence = prediction[0][index]

    st.divider()
    st.subheader(f"Prediction: **{label[2:]}**")
    st.progress(float(confidence), text=f"Confidence Score: {round(confidence*100, 2)}%")

    # --- Intelligence: Gemini Eco-Fact ---
    with st.spinner("Generating environmental insight..."):
        prompt = (f"The user has found {label[2:]} waste. Provide a 20-word "
                  f"environmental fact about this material and mention which "
                  f"colored bin it belongs in THere are 3 dustbins red , blue , green (green = wet/biodegradable waste , blue = Dry/Recyclable waste , red = Hazardous/Biomedical waste like batteries,medicines, sharp objects)  for a Green Campus.")
        try:
            response = gemini_model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.error(f"Gemini API Error: {e}")

# --- Footer ---
st.caption("Developed By AVM Internships - Modular AI Architecture")
