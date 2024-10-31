Driver Drowsiness Detection System

A real-time system designed to detect driver drowsiness using computer vision and machine learning. This project leverages a Convolutional Neural Network (CNN) and OpenCV to monitor driver alertness and trigger alerts when signs of drowsiness are detected.

📄 Project Overview

This Driver Drowsiness Detection System monitors a live video feed to assess driver alertness. The system classifies eye states (open or closed) using a trained CNN model and triggers an audio alert if prolonged eye closure is detected. This is intended to help reduce accidents caused by driver fatigue.

🔹 Features

Real-Time Detection: Monitors video feed and detects drowsiness in real-time.
Eye State Classification: CNN model trained to classify eye states (open vs. closed).
Alert Mechanism: Triggers an audio alert when drowsiness is detected.
Scoring System: Uses a scoring algorithm to track consecutive closed-eye frames, minimizing false positives.
🔹 Technologies Used

Python: Scripting and model implementation.
OpenCV: Real-time video processing and facial feature detection.
TensorFlow/Keras: Building and training the CNN model.
Haar Cascades: Detecting face and eye regions.
📂 Project Structure

graphql
Copy code
.
├── drowsinessdetection.py          # Main driver detection script
├── model.py                        # CNN model script
├── haarcascade_frontalface_alt.xml # Haar Cascade for face detection
├── haarcascade_righteye_2splits.xml
├── haarcascade_lefteye_2splits.xml
├── cnnCat2.h5                      # Pre-trained CNN model file
├── alarm.wav                       # Audio file for drowsiness alert
🔧 Installation and Setup

Clone the Repository:
bash
Copy code
git clone https://github.com/yourusername/Driver-Drowsiness-Detection.git
Install Required Libraries:
bash
Copy code
pip install -r requirements.txt
Run the Application:
bash
Copy code
python drowsinessdetection.py
Audio Alert: Ensure alarm.wav is in the project directory for the alert sound to work.
