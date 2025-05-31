# 🎙️ Speech Emotion Recognition using Deep Learning

This project implements a robust Speech Emotion Recognition (SER) system using handcrafted audio features and a 1D Convolutional Neural Network (CNN). It achieves **95.8% accuracy** across four benchmark datasets: **RAVDESS, CREMA-D, TESS,** and **SAVEE**.

---

## 🔍 Overview

Emotion in speech conveys critical context beyond words—key for applications in **virtual assistants**, **mental health monitoring**, **customer service**, and **affective computing**. This project aims to classify speech emotions using signal-processing techniques and a deep learning model optimized for sequence data.

---

## 🚀 Key Features

- ✅ **High accuracy (95.8%)** on a merged multi-dataset emotion corpus  
- 📊 **Feature-based approach** using ZCR, RMS, and MFCC (2376D vector)  
- 🧠 **Custom 1D CNN** for sequential feature learning  
- 🔄 **Audio augmentation**: noise, pitch shifting, stretching, time shifting  
- 🧪 Evaluated using confusion matrix and classification reports  

---

## 📚 Datasets Used

| Dataset   | Emotions Covered                            | Speakers       |
|-----------|---------------------------------------------|----------------|
| **RAVDESS**  | Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised | 24 (12M, 12F) |
| **CREMA-D** | Angry, Disgust, Fear, Happy, Neutral, Sad | 91 diverse     |
| **TESS**    | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise | 2F            |
| **SAVEE**   | Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise | 4M            |

Combined total: **7 emotion classes** across **121+ actors**.

---

## 🔬 Feature Extraction

Extracted using [Librosa](https://librosa.org):
- **Zero-Crossing Rate (ZCR)** – Temporal voicing cues  
- **Root Mean Square (RMS)** – Intensity measure  
- **MFCCs (13)** – Spectral envelope characteristics  

Each 2.5s audio segment → 2376D vector  
All features standardized using `StandardScaler`.

---

## 🎧 Data Augmentation

To improve generalization and handle imbalances:
- **Noise injection**  
- **Pitch shifting**
- **Time stretching**
- **Shifting (circular roll)**

Augmentations tripled the dataset size.

---

## 🧠 Model Architecture: 1D CNN

```
Input: (2376, 1)
→ Conv1D + ReLU + BatchNorm + MaxPool
→ Conv1D + ReLU + BatchNorm + MaxPool
→ Conv1D + ReLU + BatchNorm + MaxPool
→ Flatten
→ Dense(512) + ReLU + BatchNorm
→ Dense(7) + Softmax
```
* **Optimizer:** RMSprop
* **Loss:** Categorical Crossentropy
* **Callbacks:** EarlyStopping, ReduceLROnPlateau
* **Batch Size:** 64, **Epochs:** 50 (with early stop)
---

## 📈 Results

* 🎯 **Test Accuracy:** `95.8%`
* 📊 **F1-Score (macro/weighted):** \~`0.96`
* 📉 **Minimal overfitting** due to regularization
* ✅ Performs robustly across all datasets


---

## 📌 Real-World Applications

* 🧠 **Mental health monitoring** (stress, depression detection)
* 🗣️ **Emotion-aware virtual assistants** and chatbots
* 🧓 **Elderly care systems** with emotion-sensitive feedback
* 🎮 **Emotion-adaptive games and entertainment systems**



---

## 📌 Future Work

* 🔁 Real-time SER deployment using streaming inference
* 🤝 Multimodal SER (e.g. video + speech)
* 🧠 Use transformer-based audio backbones (e.g., wav2vec, YAMNet)
* 📱 Mobile optimization (TFLite / Edge deployment)

---






