import cv2
import torch
import numpy as np
from models.emotion_cnn import EmotionCNN
from utils.preprocess import get_transforms
from PIL import Image

# Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Preprocessing
transform = get_transforms()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (48,48))
    face_img = Image.fromarray(face_img)  # Convert to PIL Image
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(face_tensor)
        pred = torch.argmax(output,1).item()
    return emotion_labels[pred]

def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face_img = frame[y:y+h, x:x+w]
        emotion = predict_emotion(face_img)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    return frame
