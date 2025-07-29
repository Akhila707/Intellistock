import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from PIL import Image

# App UI
st.title("📦 IntelliStock: Predictive Refill & Smart Shelf Monitoring")
st.write("This app uses YOLOv8 for object detection and LSTM for sales prediction to assist with inventory management.")

# Load YOLOv8 model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolo.pt")

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model_best.h5")

# Load the models
yolo_model = load_yolo_model()
lstm_model = load_lstm_model()

# Predict sales
def predict_sales(input_data):
    if input_data.ndim == 2:
        input_data = input_data.reshape(1, 7, 11)
    prediction = lstm_model.predict(input_data)
    return float(prediction[0][0])

# Shelf status checker using YOLO
def check_shelf_status(results):
    class_names = results.names
    detected_classes = [class_names[int(cls)] for cls in results.boxes.cls]
    st.write("🧠 Detected classes:", detected_classes)

    if 'Empty-Space' in detected_classes:
        return "Empty"
    elif 'Reduced' in detected_classes:
        return "Reduced"
    else:
        return "Normal"

# Smart alert logic
def smart_alert_system(results, sales_data_input=None, sales_threshold=0.08):
    shelf_status = check_shelf_status(results)
    predicted_sales = predict_sales(sales_data_input) if sales_data_input is not None else None

    if shelf_status == "Empty":
        if predicted_sales and predicted_sales > sales_threshold:
            st.error(f"🚨 ALARM: Shelf is EMPTY and predicted sales = {predicted_sales:.4f}")
        else:
            st.warning("🚨 Shelf is EMPTY, but sales are low. No immediate refill needed.")
        play_alarm()

    elif shelf_status == "Reduced":
        if predicted_sales and predicted_sales > sales_threshold:
            st.warning(f"⚠️ WARNING: Shelf stock is REDUCED and predicted sales = {predicted_sales:.4f}")
            play_alarm()
        else:
            st.success("✅ Shelf is reduced but sales are manageable.")

    else:
        st.success("✅ Shelf looks fine!")

# Play alarm
def play_alarm():
    alarm_path = "alarm-siren-sound-effect-type-01-294194.mp3"
    with open(alarm_path, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

# Upload image or alarm file
uploaded_file = st.file_uploader("📤 Upload shelf image or alarm sound (.mp3)", type=["jpg", "jpeg", "png", "mp3"])

if uploaded_file:
    st.success("✅ File uploaded!")

    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Shelf Image", use_column_width=True)

        # YOLO detection
        st.write("🔍 Running YOLOv8 object detection...")
        results = yolo_model(image)[0]
        rendered_image = results.plot()
        st.image(rendered_image, caption="📌 Detection Results", use_column_width=True)

        # Dummy LSTM input (replace with real input logic later)
        dummy_sales_data = np.random.rand(1, 7, 11)
        smart_alert_system(results, sales_data_input=dummy_sales_data)

    elif uploaded_file.type == "audio/mp3":
        st.audio(uploaded_file.read(), format="audio/mp3")
        st.info("🎧 Playing uploaded sound.")