import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------
# Load your trained emotion model
# ----------------------------
model = load_model('emotion_model.h5')  # replace with your model path
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
                4: "Sad", 5: "Surprise", 6: "Neutral"}  # adjust as per your model

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Live Emotion Recognition")

# Option to use webcam
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])  # placeholder for video frames

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# Function to process each frame
# ----------------------------
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # model input size
        roi_gray = roi_gray.astype('float')/255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict emotion
        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        emotion = emotion_dict[max_index]

        # Draw rectangle and label on original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (36,255,12), 2)
    return frame

# ----------------------------
# Start webcam feed
# ----------------------------
if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror the frame
        frame = process_frame(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
