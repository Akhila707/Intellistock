import streamlit as st
import torch
from keras.models import load_model
import os

# Title
st.title("IntelliStock: Hybrid Intelligence for Predictive Refill and Smart Shelf Monitoring")
st.write("Welcome to your deployed ML app!")

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    model_path = os.path.join(os.getcwd(), "yolo.pt")
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    model_path = os.path.join(os.getcwd(), "lstm_model_best.h5")
    return load_model(model_path)

yolo_model = load_yolo_model()
lstm_model = load_lstm_model()

# Upload file
uploaded_file = st.file_uploader("Upload an image or audio", type=["jpg", "png", "jpeg", "mp3"])

if uploaded_file:
    st.success("File uploaded!")
    # Your prediction logic here based on file type...
