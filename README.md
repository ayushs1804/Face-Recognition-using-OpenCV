# Real-Time Face Recognition System using OpenCV 🚀

A real-time face recognition system built using **Python**, **OpenCV**, and the **LBPH Face Recognizer**. The project supports:

- Real-time face recognition
- Multi-user dataset collection
- Eye detection
- Smile detection
- Confidence score prediction
- Live webcam processing
- Dataset training using LBPH algorithm

This project demonstrates a complete **Computer Vision pipeline** from dataset generation to live face recognition.

## Overview

This project is a complete real-time face recognition system built using OpenCV and the LBPH (Local Binary Patterns Histograms) face recognizer. The system is capable of:

- Collecting face datasets using a webcam
- Detecting faces in real-time
- Detecting eyes and smiles
- Training a face recognition model
- Recognizing multiple users live through webcam input
- Displaying confidence scores for recognition

The project demonstrates a full computer vision pipeline from data collection to deployment-ready recognition.

---

# Features

## Face Dataset Collection

- Captures face images directly from webcam
- Automatically saves face samples into dataset folder
- Supports multiple users with unique IDs
- Collects multiple expressions and angles
- Detects:
  - Face
  - Eyes
  - Smile

## Face Recognition

- Real-time recognition using webcam
- Multi-person recognition support
- Displays:
  - Person name
  - Confidence score
  - Eye detection
  - Smile detection

## Model Training

- Uses LBPH Face Recognizer
- Generates trained model file:

```text
trainer/trainer.yml
```

---

# Technologies Used

| Technology     | Purpose                  |
| -------------- | ------------------------ |
| Python         | Programming Language     |
| OpenCV         | Computer Vision          |
| Haar Cascades  | Face/Eye/Smile Detection |
| LBPH Algorithm | Face Recognition         |
| NumPy          | Numerical Operations     |
| Pillow         | Image Processing         |

---

# Project Structure

```text
OpenCV-Face-Recognition/
│
├── FacialRecognition/
│   │
│   ├── dataset/
│   ├── trainer/
│   │
│   ├── 01_face_dataset.py
│   ├── 02_face_training.py
│   ├── 03_face_recognition.py
│   │
│   └── haarcascade files
│
├── README.md
```

---

# Working Pipeline

## Phase 1: Data Gathering

The webcam captures multiple face images of users.

The system:

- Detects face
- Detects eyes
- Detects smile
- Saves grayscale cropped face images

Dataset images are saved as:

```text
User.<ID>.<SampleNumber>.jpg
```

Example:

```text
User.1.1.jpg
User.1.2.jpg
User.2.1.jpg
```

---

## Phase 2: Training the Recognizer

The training script:

1. Reads all images from dataset folder
2. Converts images to grayscale
3. Extracts face regions
4. Reads user IDs from filenames
5. Trains the LBPH recognizer
6. Saves trained model

Generated model:

```text
trainer/trainer.yml
```

---

## Phase 3: Real-Time Recognition

The recognition system:

- Opens webcam
- Detects faces in live video
- Predicts person identity
- Displays confidence score
- Detects eyes and smile simultaneously

---

# Installation

## Clone Repository

```bash
git clone https://github.com/ayushs1804/Face-Recognition-using-OpenCV.git
```

---

## Install Dependencies

```bash
pip install opencv-contrib-python pillow numpy
```

Important:

Use:

```text
opencv-contrib-python
```

because LBPH recognizer exists only in contrib package.

---

# How to Run

## Step 1: Collect Dataset

Run:

```bash
python 01_face_dataset.py
```

Input:

```text
Enter User ID ==> 1
Enter Person Name ==> Ayush
```

The system captures face samples and stores them in:

```text
dataset/
```

---

## Step 2: Train the Model

Run:

```bash
python 02_face_training.py
```

Output:

```text
trainer/trainer.yml
```

---

## Step 3: Run Recognition

Run:

```bash
python 03_face_recognition.py
```

The webcam opens and starts real-time face recognition.

---

# Dataset Recommendations

For best performance:

- Use good lighting
- Capture different face angles
- Include different expressions
- Capture images with/without glasses
- Use at least 50 images per person

---

# Recognition Output

The recognition window displays:

- Face rectangle
- Person name
- Confidence percentage
- Eye detection
- Smile detection

Example:

```text
Ayush
92%
```

---

# LBPH Face Recognizer

LBPH (Local Binary Patterns Histograms) is a classical face recognition algorithm.

Advantages:

- Fast
- Lightweight
- Easy to train
- Works offline
- Good for beginners

Limitations:

- Sensitive to lighting
- Lower accuracy compared to deep learning
- Requires good dataset quality

---

# Applications

This project can be extended into:

- Smart Attendance System
- Face Unlock System
- Access Control
- Security Surveillance
- Visitor Management System
- Student Attendance Tracking
- Smart Home Authentication

---

# Future Scope and Advanced Improvements 🚀

This project serves as a strong foundation for building advanced AI-powered computer vision applications. While the current implementation uses classical OpenCV and LBPH-based recognition, the system can be significantly enhanced using modern deep learning approaches and real-world integrations.

---

# Future Scope

The current project can evolve into several real-world applications and enterprise-level systems.

## 1. Smart Attendance System

The face recognition system can automatically mark attendance for:

- Schools
- Colleges
- Offices
- Coaching Institutes
- Corporate Meetings

Possible Enhancements:

- Store attendance in database
- Export attendance to Excel/CSV
- Email attendance reports
- Cloud-based attendance dashboard
- Face mask detection during attendance

---

## 2. Smart Security & Surveillance System

This system can be integrated with CCTV cameras for:

- Unauthorized person detection
- Intruder alerts
- Real-time surveillance
- Restricted area monitoring
- Security authentication

Possible Enhancements:

- SMS/Email alert system
- Telegram alert integration
- Automatic screenshot capture
- Cloud storage of suspicious activity
- Multi-camera tracking

---

## 3. Face Unlock / Authentication System

The project can be expanded into a biometric authentication system similar to:

- Mobile face unlock
- Smart door lock systems
- Computer login authentication
- ATM security systems

Possible Enhancements:

- OTP verification
- Liveness detection
- Anti-spoofing detection
- Infrared camera support
- Voice + Face authentication

---

## 4. AI-Powered Smart Home System

The recognition system can identify users and personalize smart home behavior.

Examples:

- Auto room access
- Personalized lighting/music
- Smart assistant activation
- Visitor identification
- Family member recognition

---

## 5. Emotion and Mood Detection

Future versions can integrate emotion recognition models to detect:

- Happy
- Sad
- Angry
- Neutral
- Surprised
- Sleepy

Applications:

- Mental health analysis
- Student engagement monitoring
- Customer satisfaction analysis
- Human-computer interaction

##

---

# Advanced Technical Improvements

The current system uses Haar Cascades and LBPH, which are classical machine learning techniques. These can be upgraded using deep learning for significantly better accuracy and robustness.

---

## 1. Replace Haar Cascade with YOLO Face Detection

Current Limitation:

- Haar Cascade struggles in poor lighting
- False detections possible
- Less accurate at different angles

Upgrade:

Use:

- YOLOv8 Face Detection
- RetinaFace
- MTCNN

Benefits:

- Faster detection
- Better accuracy
- Real-time performance
- Better angle handling
- Better low-light detection

---

## 2. Replace LBPH with Deep Learning Embeddings

Current Limitation:

LBPH is sensitive to:

- Lighting
- Pose variations
- Blur
- Low-quality cameras

Modern Alternatives:

- FaceNet
- ArcFace
- DeepFace
- Dlib Face Recognition

Benefits:

- Much higher accuracy
- Better generalization
- Robust recognition
- State-of-the-art performance

---

## 3. Database Integration

Currently, names are stored manually in Python lists.

Future improvement:

Integrate databases such as:

- MySQL
- MongoDB
- PostgreSQL
- Firebase

Store:

- User details
- Face embeddings
- Attendance logs
- Login history
- Images

---

## 4. Web Dashboard Integration

Future versions can include:

- Flask Web App
- Streamlit Dashboard
- Django Integration

Features:

- Live webcam feed in browser
- User management panel
- Analytics dashboard
- Attendance reports
- Real-time monitoring

---

## 5. Cloud Deployment

The system can be deployed using:

- AWS
- Google Cloud
- Microsoft Azure
- Heroku
- Render

Applications:

- Remote monitoring
- Cloud-based recognition
- Multi-device access
- IoT integration

---

# Challenges and Limitations

Although the current project works well for educational purposes, there are limitations:

| Limitation            | Reason                        |
| --------------------- | ----------------------------- |
| Sensitive to lighting | LBPH + Haar Cascades          |
| Lower accuracy        | Classical ML approach         |
| No anti-spoofing      | Can be fooled by photos       |
| Performance issues    | Large datasets slow down LBPH |
| Limited scalability   | Manual user management        |

These limitations can be solved using modern deep learning methods.

---

# Why This Project is Valuable

This project demonstrates:

- Computer Vision fundamentals
- Real-time image processing
- Face recognition
