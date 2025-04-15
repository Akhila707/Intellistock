import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from IPython.display import Audio
import time

# Load Models
st.title("📦 IntelliStock: Smart Inventory Alarm System")
lstm_model = load_model('lstm_model_best.h5')
yolo_model = YOLO('yolo.pt')

# Alarm Function
def play_alarm():
    audio_file = open('alarm.mp3', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

# Detect shelf status using YOLO
def check_shelf_status(image):
    results = yolo_model(image)
    detected_classes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            detected_classes.append(r.names[cls])
    
    st.image(results[0].plot(), caption="Detected Shelf", use_column_width=True)
    if 'Empty-Space' in detected_classes:
        return "Empty"
    elif 'Reduced' in detected_classes:
        return "Reduced"
    else:
        return "Normal"

# Forecast sales using LSTM
def predict_sales(input_data):
    input_data = np.array(input_data).reshape(1, 7, 11)
    prediction = lstm_model.predict(input_data)
    return float(prediction[0][0])

# Upload Inputs
uploaded_image = st.file_uploader("📷 Upload Shelf Image", type=['jpg', 'jpeg', 'png'])
sample_input = st.file_uploader("📊 Upload Sales Data (npy format)", type=['npy'])

if uploaded_image and sample_input:
    image = Image.open(uploaded_image)
    sales_data = np.load(sample_input)

    # Run smart logic
    st.subheader("🔍 Analyzing Shelf...")
    shelf_status = check_shelf_status(image)
    predicted_sales = predict_sales(sales_data)

    st.write(f"**Predicted Sales:** {predicted_sales:.2f}")
    st.write(f"**Shelf Status:** {shelf_status}")

    if shelf_status == "Empty" and predicted_sales > 50:
        st.error("🚨 ALARM! Refill shelf urgently!")
        play_alarm()
    elif shelf_status == "Reduced" and predicted_sales > 50:
        st.warning("⚠ Warning: Top up stock soon.")
        play_alarm()
    elif shelf_status == "Empty":
        st.info("😌 Empty but demand is low. No rush.")
    else:
        st.success("✅ Shelf looks fine.")

