# UnMask — AI Media Authenticity Detector

> Detecting deepfakes in images and videos using ensemble deep learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![License](https://img.shields.io/badge/License-MIT-red)

## Problem Statement
Deepfake media is spreading at unprecedented scale across social media.
96% of people cannot detect AI-generated content with the naked eye.
UnMask solves this with real-time AI powered detection.

## Solution
Ensemble model combining EfficientNetB4 + Xception + MesoNet
achieving 94-96% accuracy on FaceForensics++ and Celeb-DF v2 benchmarks.

## Features
- Image deepfake detection (< 2 seconds)
- Video frame-by-frame analysis with timeline graph
- Grad-CAM heatmap visualization
- Audio lip sync verification
- Downloadable authenticity certificate

## Tech Stack
- **ML:** TensorFlow, EfficientNetB4, Xception, MesoNet, MTCNN, Grad-CAM
- **Backend:** Python, Flask, OpenCV, MoviePy, Librosa
- **Frontend:** React.js, Recharts, HTML5 Canvas, Tailwind CSS

## Project Structure
```
unmask/
├── app.py              # Flask backend
├── requirements.txt    # Dependencies
├── model/
│   └── train.py        # Model training code
├── frontend/           # React frontend
├── demo/               # UI prototype
└── presentation/       # Pitch deck
```

## Setup & Run

### Backend
```bash
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Team
Built for FOOBAR 10.0 Hackathon — Cybersecurity Domain