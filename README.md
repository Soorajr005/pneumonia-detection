# 🩺 Pneumonia Detection from Chest X-Ray using Deep Learning

This project focuses on detecting **Pneumonia** from Chest X-ray images using a Deep Learning model built with **PyTorch** and deployed using **FastAPI**.

---

## 🚀 Project Overview

Pneumonia is a serious lung infection that can be diagnosed using Chest X-rays.  
In this project, I used a pretrained **ResNet18** model (Transfer Learning) to classify X-ray images into:

- ✅ Normal
- ❌ Pneumonia

📊 **Model Test Accuracy: 80.77%**

This project demonstrates a complete ML pipeline:
- Data preprocessing  
- Model training  
- Model evaluation  
- API deployment  

---

## 🧠 Model Details

- Pretrained **ResNet18**
- Modified final fully connected layer for binary classification
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Framework: PyTorch
- Image resizing and normalization applied

---

## 📂 Dataset

- Source: Kaggle Chest X-ray Dataset
- Classes: Normal & Pneumonia
- Dataset split into training, validation, and test sets

---

## 🌐 API Deployment

The trained model is deployed using **FastAPI**.

### Features:
- Upload Chest X-ray image
- Get prediction (Normal / Pneumonia)
- Interactive API documentation using Swagger UI

### Access Swagger UI:
http://127.0.0.1:8000/docs

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- NumPy
- Pillow

---


## 📁 Project Structure

pneumonia-detection/
│
├── app/
│   └── main.py
│
├── model/
│   └── resnet18_model.pth
│
├── requirements.txt
├── README.md
└── .gitignore



## ▶️ How to Run Locally

1. Clone the repository
git clone https://github.com/Soorajr005/pneumonia-detection.git

2. Install dependencies
pip install -r requirements.txt

3. Run FastAPI server
uvicorn app.main:app --reload

4. Open in browser
http://127.0.0.1:8000/docs

