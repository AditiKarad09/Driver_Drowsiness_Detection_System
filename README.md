# Driver Drowsiness Detection System

A real-time system designed to detect driver drowsiness using computer vision and machine learning. This project leverages a Convolutional Neural Network (CNN) and OpenCV to monitor driver alertness and trigger alerts when signs of drowsiness are detected.

# 📄 Project Overview

This Driver Drowsiness Detection System monitors a live video feed to assess driver alertness. The system classifies eye states (open or closed) using a trained CNN model and triggers an audio alert if prolonged eye closure is detected. This is intended to help reduce accidents caused by driver fatigue.

# 🔹 Features

Real-Time Detection: Monitors video feed and detects drowsiness in real-time.
Eye State Classification: CNN model trained to classify eye states (open vs. closed).
Alert Mechanism: Triggers an audio alert when drowsiness is detected.
Scoring System: Uses a scoring algorithm to track consecutive closed-eye frames, minimizing false positives.

# 🔹 Technologies Used

Python: Scripting and model implementation.
OpenCV: Real-time video processing and facial feature detection.
TensorFlow/Keras: Building and training the CNN model.
Haar Cascades: Detecting face and eye regions.

# 📊 Results

Detection Accuracy: Achieves 93% accuracy in eye state classification.
Real-Time Capability: Capable of real-time processing with minimal lag, suitable for real-world applications.

