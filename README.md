# Dystopian Emotion Analysis

This Python script implements a real-time emotion recognition system with a dystopian theme. It uses a webcam to capture video, detects faces, analyzes emotions, and displays the results in an engaging, dystopian-style interface.

## Key Features

- Real-time face detection using MediaPipe
- Emotion recognition using a TensorFlow Lite model
- Dystopian-themed UI with emotion auras and "Emotional Decorum" badges
- Global emotion tracking and display
- Background image integration for immersive experience

## Technologies Used

- OpenCV for image processing and display
- TensorFlow Lite for emotion recognition
- MediaPipe for face detection
- NumPy for numerical operations

## How it works

1. Captures video from a webcam
2. Detects faces in each frame
3. Analyzes emotions for each detected face
4. Generates visual effects like emotion auras and badges
5. Displays results on a dystopian-themed background

This project creates an interactive, thought-provoking experience that explores the concept of emotion monitoring in a fictional dystopian society.

## Initialization on Raspberry Pi 5

For the Raspberry Pi, you need to create a virtual environment:
- python3 -m venv emotion_env
- source emotion_env/bin/activate
  
## Installation
To install the necessary dependencies, use the provided requirements_raspi.txt file:

Make sure you have Python installed on your Raspberry Pi.
Open a terminal and navigate to the project directory.
Activate the virtual environment as shown above.
Run the following command:
- pip install -r requirements_raspi.txt
  
This will install all the required libraries to run the script.

## Deployment

On Raspberry Pi 5 to enable the systemd service:
- git pull origin main
- sudo cp artboot.service /etc/systemd/system/artboot.service
- sudo systemctl daemon-reload
- sudo systemctl enable artboot.service
- sudo reboot

In case of errors inspect:
- sudo systemctl status artboot.service
- sudo journalctl -u artboot.service -f
