import streamlit as st
import torch
from keras.models import load_model
import os
from PIL import Image
from ultralytics import YOLO

# App Title and Description
st.title("📦 IntelliStock: Predictive Refill & Smart Shelf Monitoring")
st.write("This app uses YOLOv5 for object detection and LSTM for sales prediction to assist with inventory management.")

# Load YOLOv5 model
@st.cache_resource
def load_yolo_model():
    model_path = os.path.join(os.getcwd(), "yolo.pt")
    return YOLO(model_path)

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    model_path = os.path.join(os.getcwd(), "lstm_model_best.h5")
    return load_model(model_path)

# Load the models
yolo_model = load_yolo_model()
lstm_model = load_lstm_model()

# Upload image or audio
uploaded_file = st.file_uploader("📤 Upload shelf image or alarm sound (.mp3)", type=["jpg", "jpeg", "png", "mp3"])

if uploaded_file:
    st.success("✅ File uploaded!")

    # If it's an image
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Shelf Image", use_column_width=True)

        # Run YOLO object detection
        st.write("🔍 Running YOLOv5 object detection...")
        results = yolo_model(image)
        results.print()
        st.image(results.render()[0], caption="📌 Detection Results", use_column_width=True)

        # Play alarm if Empty-Space or Reduced is detected (optional logic)
        classes = results.names
        detected = [classes[int(cls)] for *_, cls, conf in results.xyxy[0]]
        st.write("Detected classes:", detected)

        if "Empty-Space" in detected:
            st.error("🚨 ALARM: Shelf is EMPTY!")
            with open("alarm-siren-sound-effect-type-01-294194.mp3", "rb") as f:
                alarm = f.read()
                st.audio(alarm, format="audio/mp3")

        elif "Reduced" in detected:
            st.warning("⚠️ WARNING: Shelf stock is REDUCED!")
            with open("alarm-siren-sound-effect-type-01-294194.mp3", "rb") as f:
                alarm = f.read()
                st.audio(alarm, format="audio/mp3")

        else:
            st.success("✅ Shelf looks fine!")

    # If it's audio
    elif uploaded_file.type == "audio/mp3":
        st.audio(uploaded_file.read(), format="audio/mp3")
        st.info("🎧 Playing uploaded sound.")
