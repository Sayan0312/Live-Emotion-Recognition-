# 🎥 Real-Time Emotion Detection using Facial Expressions

This project is a **Deep Learning-based real-time emotion detection system** that uses a **live camera feed** to recognize facial emotions.
The model is trained using a **Convolutional Neural Network (CNN)** and deployed as a **Streamlit web application** with webcam support.

---

## 🚀 Features

* 🎥 Real-time emotion detection using live camera
* 🧠 Deep Learning CNN model
* 🌐 Interactive Streamlit web app
* 📷 Face detection + emotion classification
* ⚡ Runs locally with webcam access

---

## 🧠 Emotions Detected

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

*(Stress-level prediction is intentionally ignored in the current version)*

---

## 🗂 Project Structure

```
emotion_app/
│── app.py                  # Streamlit app (live camera)
│── emotion_model.h5         # Trained CNN model
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 🛠 Tech Stack

* Python 3.10
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* Pillow

---

## ⚙️ How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-link>
cd emotion_app
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

> ⚠️ **Note:** Allow camera access when prompted by your browser or system.

---

## 📸 How It Works

1. Webcam captures live video frames
2. Face is detected using OpenCV
3. Face is resized and preprocessed
4. CNN model predicts the emotion
5. Emotion label is displayed in real time

---

## 📊 Model Details

* Input size: 48×48 grayscale images
* Architecture: CNN
* Optimizer: Adam
* Loss function: Categorical Crossentropy

---

## 📌 Future Enhancements

* Stress level estimation from emotions
* Multi-face detection
* Improved accuracy using transfer learning
* Cloud deployment (Streamlit Cloud)
* Mobile/web camera optimization

---

## 👨‍💻 Author

**Sayan Rana**
  Deep Learning Enthusiast

---

⭐ If you find this project useful, give it a star on GitHub!
