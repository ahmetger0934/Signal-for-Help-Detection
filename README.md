# Signal for Help Detection

🚨 **A real-time AI-powered system to detect the "Signal for Help" gesture and report emergencies.**

## Description

This project leverages AI, machine learning, and computer vision to detect the "Signal for Help" gesture in real-time video feeds. It identifies the gesture, detects faces, and reports incidents to authorities with precise location details.

## Features

- **Real-time gesture recognition** using a trained AI model with TensorFlow and Keras.
- **Hand detection** powered by Mediapipe and OpenCV.
- **Location tracking** via Google Maps API.
- **Automated reporting** using smtplib for email notifications.
- **Data augmentation and preprocessing** using scikit-learn and advanced image processing techniques.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmetger0934/Signal-for-Help-Detection.git

2.	Navigate to the project directory:
    ```bash
    cd Signal-for-Help-Detection

3.	Install the required packages:
    ```bash
    pip install -r requirements.txt
    
4.	Download the pre-trained model and place it in the models directory.

   
   **Usage**
   
   To run the program, use the following command:
   ```bash
   python detect_signal_for_help.py


