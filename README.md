# Emotion Detection using PyTorch & OpenCV

## Project Overview

This project implements a **real-time emotion detection system** capable of identifying human emotions from facial expressions using a webcam or pre-collected images. The system leverages **PyTorch** to build a Convolutional Neural Network (CNN) for emotion classification and **OpenCV** for face detection and real-time video processing.

The model is trained to classify the following seven emotions:

**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**

This system can be used for educational purposes, human-computer interaction, and as a base for more advanced emotion recognition applications.

---

## Features

- Real-time webcam emotion detection with live labels  
- CNN model trained on facial images for accurate emotion classification  
- Modular design with separate scripts for training, evaluation, and real-time detection  
- Supports custom datasets with a folder-based structure  

---

## Dataset

The project uses an **image-based dataset** organized as follows:


Each subfolder contains facial images corresponding to a specific emotion. This structure is compatible with PyTorch’s `ImageFolder` class for easy loading and preprocessing.

---

## Project Workflow

1. **Data Preprocessing**
   - Resize all images to 48x48 pixels  
   - Convert images to grayscale  
   - Normalize pixel values to [0, 1]  
   - Optional data augmentation to improve model robustness  

2. **Model Architecture**
   - CNN with 3 convolutional layers, max pooling, dropout, and fully connected layers  
   - Output layer with 7 neurons corresponding to the 7 emotion classes  

3. **Training**
   - Train the CNN on the training dataset using **Cross-Entropy Loss** and **Adam optimizer**  
   - Hyperparameters such as learning rate, batch size, and number of epochs can be modified in `train.py`  
   - Save the trained model as `emotion_model.pth`  

4. **Evaluation**
   - Evaluate the trained model on the test dataset  
   - Generate classification reports including precision, recall, and F1-score for each emotion  

5. **Real-Time Emotion Detection**
   - Detect faces in webcam frames using OpenCV’s Haar Cascade classifier  
   - Preprocess each detected face and feed it into the trained CNN model  
   - Draw rectangles around faces and display predicted emotion labels in real-time  

---

## Technologies Used
1. Python: Main programming language

2. PyTorch: For building and training the CNN model

3. OpenCV: Face detection and real-time webcam processing

4. NumPy: Numerical computations

---

## Future Enhancements
1. Develop a GUI interface for better user experience

2. Integrate advanced face detectors like MTCNN or Dlib for improved accuracy

3. Implement emotion tracking over time to analyze emotional trends

4. Deploy as a web application using Flask or FastAPI

5. Apply data augmentation to improve model generalization and robustness
