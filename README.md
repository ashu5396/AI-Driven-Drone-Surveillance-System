# AI-Driven Real-Time CCTV Surveillance System for Drone Detection and Anomaly Monitoring

## Overview

This project presents an intelligent CCTV surveillance system capable of detecting drones, tracking their movement, identifying abnormal behavior, and generating real-time alerts. The system combines deep learning, computer vision, and web technologies to provide continuous automated surveillance for restricted environments.

The proposed framework integrates YOLOv8 for drone detection, ByteTrack for multi-object tracking, and an LSTM Autoencoder for temporal anomaly detection. A Flask-based dashboard enables live monitoring, event management, and playback of recorded alert videos.

---

## Key Features

- Real-time drone detection using YOLOv8
- Multi-object tracking using ByteTrack
- Polygon-based restricted zone monitoring
- Hover detection
- Speed estimation
- Temporal anomaly detection using LSTM Autoencoder
- SQLite event logging
- Automatic alert video recording
- Dynamic camera addition and removal
- Interactive web dashboard
- Socket.IO based live streaming

---

## System Architecture

```
Camera
   │
   ▼
Frame Capture
   │
   ▼
YOLOv8 Drone Detection
   │
   ▼
ByteTrack Tracking
   │
   ▼
Behavior Feature Extraction
   │
   ▼
LSTM Autoencoder
   │
   ▼
Anomaly Detection
   │
   ▼
SQLite Database
   │
   ▼
Flask Dashboard
```

---

## Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Backend |
| YOLOv8 | Drone Detection |
| ByteTrack | Multi-object Tracking |
| PyTorch | Deep Learning |
| OpenCV | Image Processing |
| Flask | Web Framework |
| Flask-SocketIO | Real-time Streaming |
| SQLite | Event Storage |
| FFmpeg | Alert Video Recording |
| HTML/CSS/JavaScript | Dashboard |

---

## Project Structure

```
models/
script/
videos/
screenshots/
requirements.txt
README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/ashu5396/AI-Driven-Drone-Surveillance-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python script/app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## Dashboard

### Home Page

(Add Screenshot Here)

---

### Drone Detection

(Add Screenshot Here)

---

### Restricted Zone Monitoring

(Add Screenshot Here)

---

### Alert Playback

(Add Screenshot Here)

---

## Experimental Results

The proposed system demonstrates reliable real-time surveillance performance by integrating drone detection, object tracking, behavioral analysis, and anomaly detection into a unified framework.

Performance evaluation includes:

- LSTM Training Loss
- Reconstruction Error Distribution
- Threshold Estimation
- Normal vs Anomalous Classification
- Reconstruction Error Trend

---

## Future Work

- Multi-camera distributed deployment
- Edge AI optimization
- Thermal camera integration
- Drone re-identification
- Cloud-based surveillance
- Mobile application support

---

## Author

**Ashutosh Kumar**

M.Sc. Artificial Intelligence

Central University of South Bihar

---

## License

This project is intended for academic and research purposes.
