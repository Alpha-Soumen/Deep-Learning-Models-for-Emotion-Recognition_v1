
# 🎭 Deep Learning Models for Emotion Recognition

This repository contains an end-to-end deep learning solution for **emotion detection from facial images**, built using TensorFlow/Keras and deployed using **Gradio** for an interactive web interface.

> **Notebook Type**: Jupyter Notebook (.ipynb)
> **Execution Environment**: Google Colab (Free Tier T4 GPU)
> **Runtime Strategy**: Models were trained sequentially – first two custom CNN models, followed by VGG16 and then ResNet50
> **Training**: All models trained for up to 50 epochs (early stopping applied)
> **Deployment**: Final deployment used the ResNet50 model with face detection and confidence visualization

---

## 🔍 Project Overview

The goal of this project is to detect one of **seven emotions** — Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral — from facial images using deep learning. The system is built and evaluated using multiple models, and the best-performing model is deployed via a user-friendly Gradio interface.

---

## 📁 Contents

* `Emotion_Detection.ipynb`: Core notebook with all models, training, evaluation, and deployment logic
* `/models/`: Saved `.keras` and `.h5` model weights
* `/images/`: Visual outputs (confusion matrices, confidence plots)
* `/gradio_app/`: Gradio UI source code

---

## 🧠 Models Used

### 1. **Custom CNN From Scratch**

* **Architecture**: 6 Convolutional layers, BatchNorm, MaxPooling, Dropout, Dense(1024), Softmax(7)
* **Input**: Grayscale 48×48 images
* **Purpose**: Baseline model to learn emotion features from scratch
* **Performance**:

  * Train Accuracy: 82.00%
  * Validation Accuracy: **61.52%**
  * Weak F1-score balance, especially poor in 'Disgust' category

---

### 2. **Custom CNN With Augmentation**

* **Same architecture** as above + **Image Augmentation**
* **Techniques**: Rotation, shift, zoom, shear, horizontal flip
* **Performance**:

  * Train Accuracy: 54.76%
  * Validation Accuracy: **56.02%**
  * Better generalization but lower accuracy

---

### 3. **VGG16 Transfer Learning**

* **Pre-trained VGG16 (ImageNet)** with custom classification head
* **Frozen base layers**, new top classifier layers
* **Input**: RGB 224×224 images
* **Performance**:

  * Validation Accuracy: **59.52%**
  * Reasonable class balance but slightly lower overall performance

---

### 4. **ResNet50 Transfer Learning** ✅ **(Deployed Model)**

* **ResNet50V2 base**, with last 50 layers unfrozen
* **Custom head**: Dropout, BatchNorm, Dense(64), Softmax(7)
* **Input**: RGB 224×224 images
* **Class Weights**: Used to address class imbalance
* **Performance**:

  * Validation Accuracy: **56.80%**
  * Highest weighted F1-score: **0.57**
  * Best per-class results for **Happy (F1: 0.82)** and **Surprise (F1: 0.70)**
  * Best **AUC** for Disgust (0.90)

---

## 🏆 Best Model for Production

✅ **ResNet50 Transfer Learning** was chosen for production due to:

* Strong balanced performance across emotions
* Robust handling of real-world face data (thanks to transfer learning)
* Better interpretability and ROC metrics
* Integrated face detection and preprocessing pipeline

---

## 🖥️ Web Interface (Gradio)

### ▶ Screenshot:

![Screenshot 2025-05-10 125158](https://github.com/user-attachments/assets/7274667f-d5c4-4e84-b2f8-7688a10f9af0)
### 🔧 Features:

* Real-time face detection via **OpenCV Haar Cascade**
* Automatic face cropping and resizing to 224×224
* Emotion prediction with confidence score
* Confidence bar chart for class probabilities
* Webcam & image upload support


---

## 🧪 Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score (per class)
* Confusion Matrix
* ROC-AUC curves
* Training history plots

All model results and outputs are included in the original `.ipynb` notebook.

---

## ✅ Code Execution Summary

* ✅ All code cells executed successfully on **Google Colab** with **T4 GPU**
* ✅ No major runtime errors
* 🚫 ResNet50 model showed a brief spike in `val_loss`, but recovered

---


## 🚀 Future Improvements

* Implement **Grad-CAM** visualizations for model interpretability
* Convert model to **TFLite** for edge/mobile deployment
* Integrate REST API via **FastAPI**
* Containerize with **Docker** and deploy to **Kubernetes**
* Apply **mixed precision** training for performance boost

---

## 📷 Output Preview
![Screenshot 2025-05-10 125143](https://github.com/user-attachments/assets/634cd9bc-6f1b-44c8-9b51-49f4f3bd830b)


---

## 📌 How to Run

1. Clone this repo and open the notebook in **Google Colab**
2. Upload your `kaggle.json` (do NOT print it)
3. Run all cells sequentially (Models → Evaluation → Gradio)
4. Or simply use the Gradio interface at the end

---


## 🙏 Acknowledgments

* [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
* TensorFlow/Keras team
* Gradio for the interactive UI framework

---

## 📄 Notebook PDF

You can download or view the full notebook in PDF format here:

👉 [Download Emotion Detection Notebook (PDF)] [EDUD_1.pdf](https://github.com/user-attachments/files/20147006/EDUD_1.pdf)


> 📌 Note: All models were trained using Google Colab’s free T4 GPU.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Developed by

**Soumen Bhunia**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/soumen-bhunia/)

---

