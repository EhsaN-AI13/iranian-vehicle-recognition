# Iranian Vehicle Recognition 🚗🇮🇷

An end-to-end deep learning system for recognizing Iranian vehicle types from images.

The project uses **PyTorch + ResNet18 + Transfer Learning**, exposes predictions through a **FastAPI REST API**, packages the application with **Docker**, and is deployed publicly on **Render**.

## 🌐 Live Demo

**Try the deployed web app:**  
https://iranian-vehicle-recognition.onrender.com

Upload a vehicle image and the model returns the top predicted classes with confidence scores.

## 🎯 Project Overview

The goal is to build a practical computer vision pipeline capable of classifying Iranian vehicles across **29 classes**.

The project covers the complete workflow:

**Dataset → Data Preparation → Transfer Learning → Fine-Tuning → Evaluation → Prediction → FastAPI → Docker → Cloud Deployment**

## 🧠 Model

- Architecture: **ResNet18**
- Framework: **PyTorch**
- Strategy: **Transfer Learning + Fine-Tuning**
- Input size: **224 × 224**
- Number of classes: **29**
- Inference: **CPU**
- Pretrained weights: ImageNet

During fine-tuning, the final ResNet layer was replaced with a 29-class classifier and the deeper `layer4` block was unfrozen.

## 📊 Dataset

The project uses the **SIVD (Iranian Vehicles Dataset)**.

Training data used locally:

- **29,363 images**
- **29 vehicle classes**
- Stratified train/validation split
- 80% training
- 20% validation

Vehicle categories include examples such as:

- Peugeot 405
- Peugeot Pars
- Samand
- Dena
- Pride variants
- Tiba
- Saina
- Quik
- Shahin
- Runna
- Xantia
- MVM models
- and other Iranian-market vehicles

## 📈 Evaluation

Validation performance of the fine-tuned ResNet18 model:

| Metric | Result |
|---|---:|
| Validation Accuracy | **68.81%** |
| Macro Precision | **73%** |
| Macro Recall | **68%** |
| Macro F1 | **67%** |
| Weighted F1 | **70%** |

The validation set contains **5,873 images**.

Performance varies by vehicle class, with visually similar models being the main source of confusion.

## 🔍 Example Prediction

A real test image was correctly classified as:

**Peugeot 405 — 99%+ confidence**

The same model is also used by the public deployment.

## 🏗️ Architecture

```text
                 Vehicle Image
                       │
                       ▼
                Web Frontend
                       │
                       ▼
                  FastAPI API
                       │
                       ▼
              VehiclePredictor
                       │
                       ▼
                PyTorch ResNet18
                       │
                       ▼
                29-Class Output
                       │
                       ▼
          Top-3 Predictions + Confidence
```

## 🚀 API

### Health Check

```
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Vehicle Prediction

```
POST /predict
```

Send an image using multipart form-data with the field:

```
file
```

Example response:

```json
{
  "filename": "car5.jpg",
  "predictions": [
    {
      "class": "405",
      "confidence": 0.99996
    }
  ]
}
```

## 🐳 Docker

The application is containerized using Docker.

The production image uses CPU-only PyTorch to keep the deployment independent of NVIDIA CUDA hardware.

Build:

```bash
docker build -t iranian-vehicle-api .
```

Run:

```bash
docker run --rm -p 8000:8000 iranian-vehicle-api
```

## 🧪 Testing

The project includes API and prediction tests.

Current API test suite:

```text
3 passed
```

Tests cover:

- Root endpoint
- Health endpoint
- Real prediction endpoint
- Prediction response structure

## 📁 Project Structure

```text
iranian-vehicle-recognition/
│
├── api.py
├── predictor.py
├── train.py
├── evaluate.py
├── predict.py
├── predict_visual.py
│
├── src/
│   ├── dataset.py
│   └── model.py
│
├── models/
│   └── finetune_best_model.pth
│
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── requirements.txt
├── requirements-prod.txt
├── .dockerignore
└── .gitignore
```

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- ResNet18
- Transfer Learning
- Computer Vision
- FastAPI
- Docker
- Render
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- Pytest
- Git & GitHub

## 💡 What This Project Demonstrates

This project demonstrates practical experience with:

- Image classification
- Transfer learning
- Fine-tuning pretrained CNNs
- Dataset preparation
- Stratified train/validation splitting
- Model evaluation
- Top-K prediction
- Confidence scores
- REST API development
- FastAPI file uploads
- Docker containerization
- CPU-based PyTorch inference
- Automated testing
- Cloud deployment
- Building an end-to-end AI product

## ⚠️ Limitations

The current validation accuracy is **68.81%**, so the model should not be considered production-grade for safety-critical or high-stakes vehicle identification.

The largest challenges are visually similar vehicle classes and class imbalance.

Future improvements could include:

- Stronger data augmentation
- Class-balanced sampling
- More extensive fine-tuning
- Higher-resolution inputs
- Efficient pretrained architectures
- Confusion-driven dataset improvement
- Additional real-world test images
- Model calibration
- Monitoring and logging

## 👨‍💻 Author

**Ehsan Ghasemi**

GitHub: https://github.com/EhSaN-AI13

---

⭐ If you find the project interesting, feel free to explore the code and try the live demo.
