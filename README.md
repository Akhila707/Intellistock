<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=IntelliStock&fontSize=70&fontColor=ffffff&fontAlignY=38&desc=Predictive%20Inventory%20Management%20System&descSize=18&descColor=a8b2d8&descAlignY=58&animation=fadeIn"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Computer%20Vision-00FFFF?style=for-the-badge&logo=opencv&logoColor=black"/>
  <img src="https://img.shields.io/badge/LSTM-Time%20Series-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Conference-ICNCS%202025-ff6b6b?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/MAE-0.0020-brightgreen?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20Ultralytics-orange?style=flat-square"/>
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"/>
</p>

> **🏆 Presented at ICNCS 2025, VIT Chennai**
> *Hybrid LSTM + YOLOv8 system for predictive demand forecasting and real-time smart shelf monitoring*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Model Details](#-model-details)
- [Results](#-results)
- [Pipeline Workflow](#-pipeline-workflow)
- [Conference](#-conference)
- [Author](#-author)

---

## 🧠 Overview

**IntelliStock** is an intelligent, hybrid inventory management system that combines the power of **deep learning-based time series forecasting (LSTM/BiLSTM)** with **real-time computer vision (YOLOv8)** to automate stock monitoring and predict demand before shortages occur.

Traditional inventory systems are reactive — they restock *after* products run out. IntelliStock flips this model by:
- 📷 **Seeing** shelf stock levels in real-time via camera feeds using YOLOv8
- 📈 **Predicting** future demand using LSTM-based sequence modeling
- 🔔 **Alerting** when to reorder, before shelves go empty

This makes it ideal for **retail stores, warehouses, supermarkets, and smart supply chains**.

---

## ✨ Key Features

- 🎯 **Hybrid AI Pipeline** — Seamlessly integrates YOLOv8 object detection with LSTM demand forecasting
- 📊 **Predictive Refill Alerts** — Forecasts stock depletion before it happens
- 🖼️ **Image-Driven Inventory Tracking** — Real shelf images → automatic item counts
- ⚡ **Scalable Batch Pipelines** — Efficiently processes large-scale image and time-series data
- 📉 **Ultra-Low Forecasting Error** — Achieved **MAE = 0.0020** on test data
- 🔄 **End-to-End Automation** — From image capture to reorder recommendation, fully automated
- 📦 **Modular Design** — Vision and forecasting modules can be used independently

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    IntelliStock Pipeline                 │
│                                                         │
│  📷 Camera / Image Feed                                  │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────┐     ┌───────────────────────────┐  │
│  │   YOLOv8 Module │────▶│  Stock Count Extractor    │  │
│  │  (Object Det.)  │     │  (ROI-based Counting)     │  │
│  └─────────────────┘     └───────────┬───────────────┘  │
│                                      │                  │
│                          ┌───────────▼───────────────┐  │
│                          │   Time Series Database     │  │
│                          │   (Stock Level History)    │  │
│                          └───────────┬───────────────┘  │
│                                      │                  │
│                          ┌───────────▼───────────────┐  │
│                          │   LSTM / BiLSTM Forecaster │  │
│                          │   (Demand Prediction)      │  │
│                          └───────────┬───────────────┘  │
│                                      │                  │
│                          ┌───────────▼───────────────┐  │
│                          │   Alert & Reorder Engine   │  │
│                          │   📦 Refill Recommendation │  │
│                          └───────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Computer Vision** | YOLOv8 (Ultralytics), OpenCV |
| **Deep Learning** | PyTorch, LSTM, BiLSTM |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML Utilities** | Scikit-learn |
| **Deployment** | Flask / Streamlit |
| **Version Control** | Git, GitHub |
| **Language** | Python 3.9+ |

---

## 📁 Project Structure

```
IntelliStock/
│
├── 📂 data/
│   ├── raw/                    # Raw image datasets & sales records
│   ├── processed/              # Cleaned & preprocessed data
│   └── time_series/            # Stock level time series data
│
├── 📂 models/
│   ├── yolov8/                 # YOLOv8 weights & config
│   │   ├── best.pt             # Trained YOLOv8 model
│   │   └── data.yaml           # Dataset config
│   └── lstm/
│       ├── lstm_model.pth      # Trained LSTM weights
│       └── bilstm_model.pth    # Trained BiLSTM weights
│
├── 📂 src/
│   ├── detection/
│   │   ├── detect_shelf.py     # YOLOv8 shelf detection
│   │   └── roi_extractor.py    # ROI-based stock counter
│   ├── forecasting/
│   │   ├── lstm_model.py       # LSTM architecture
│   │   ├── bilstm_model.py     # BiLSTM architecture
│   │   └── train.py            # Training script
│   ├── pipeline/
│   │   ├── batch_pipeline.py   # Scalable batch processing
│   │   └── alert_engine.py     # Reorder alert system
│   └── utils/
│       ├── preprocess.py       # Data preprocessing
│       └── visualize.py        # Plotting & dashboards
│
├── 📂 notebooks/
│   ├── EDA.ipynb               # Exploratory data analysis
│   ├── LSTM_Training.ipynb     # Model training walkthrough
│   └── YOLOv8_Training.ipynb   # Vision model training
│
├── 📂 results/
│   ├── metrics/                # MAE, RMSE, accuracy logs
│   └── plots/                  # Forecast charts & detection outputs
│
├── app.py                      # Streamlit / Flask app entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.9
pip
Git
```

### 1. Clone the Repository

```bash
git clone https://github.com/Akhila707/Intellistock.git
cd Intellistock
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Object Detection Module

```bash
python src/detection/detect_shelf.py --source data/raw/shelf_images/
```

### 5. Train LSTM Forecasting Model

```bash
python src/forecasting/train.py --epochs 100 --model lstm
```

### 6. Run the Full Pipeline

```bash
python src/pipeline/batch_pipeline.py
```

### 7. Launch Dashboard

```bash
streamlit run app.py
```

---

## 🤖 Model Details

### 🎯 YOLOv8 — Shelf Object Detection

| Parameter | Value |
|---|---|
| Base Model | YOLOv8n / YOLOv8s |
| Task | Object Detection |
| Input | Shelf camera images |
| Output | Bounding boxes + item counts |
| Training Data | Custom shelf image dataset |

- Detects individual products on retail shelves
- Counts items per shelf zone using ROI (Region of Interest) extraction
- Feeds real-time stock count data into the forecasting pipeline

---

### 📈 LSTM / BiLSTM — Demand Forecasting

| Parameter | Value |
|---|---|
| Architecture | LSTM / Bidirectional LSTM |
| Input | Sequential stock level history |
| Output | Predicted future demand |
| Loss Function | Mean Absolute Error (MAE) |
| **Test MAE** | **0.0020** |
| Optimizer | Adam |

- Learns temporal patterns in stock depletion
- BiLSTM captures both forward and backward dependencies in demand patterns
- Outputs future demand values to trigger reorder alerts

---

## 📊 Results

| Metric | Value |
|---|---|
| **MAE (Mean Absolute Error)** | **0.0020** |
| Detection Accuracy | High precision on shelf items |
| Pipeline Scalability | Batch processing ready |
| Alert Lead Time | Configurable (hours / days ahead) |

<div align="center">

> 📉 *MAE of 0.0020 achieved on demand forecasting test set — indicating near-perfect stock prediction accuracy.*

</div>

---

## 🔄 Pipeline Workflow

```
1. 📷  Capture shelf images (live camera / batch upload)
        │
2. 🔍  YOLOv8 detects & counts products per shelf zone
        │
3. 📥  Stock counts logged to time series database
        │
4. 🧠  LSTM/BiLSTM forecasts next N-day demand
        │
5. ⚠️  Alert engine checks forecast vs. reorder threshold
        │
6. 📦  Reorder recommendation generated automatically
```

---

## 🏆 Conference

This project was presented at:

> **ICNCS 2025** — International Conference on Networks and Communication Systems
> **Venue:** VIT Chennai
> **Paper:** *IntelliStock — Predictive Refill & Smart Shelf Monitoring*
> **Author:** PV Akhila

---

## 👩‍💻 Author

**PV Akhila**
*Data Scientist | AI & ML Researcher | Physics Background*

---

<div align="center">

⭐ *If you find this project useful, please consider giving it a star!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer"/>

</div>
