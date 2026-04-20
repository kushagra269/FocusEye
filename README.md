# FocusEye

An AI-powered procrastination blocker that uses your webcam to detect when you're distracted — looking away, turning your head, or using your phone — and blocks distracting websites until you re-focus.

---

## How It Works

FocusEye runs a real-time computer vision pipeline on your webcam feed using three detection systems:

| Signal | Technology | What It Detects |
|--------|-----------|-----------------|
| Eye gaze | MediaPipe FaceMesh (478 landmarks) | Iris position deviating from screen |
| Head pose | OpenCV `solvePnP` | Yaw/pitch exceeding ±20°/±22° from baseline |
| Phone use | YOLOv8-nano + MediaPipe Hands + Pose | Phone in frame, grip, or forward-head posture |

When distraction is detected for ~0.67 seconds (20 consecutive frames), the state machine logs a **distraction episode**. When cumulative away time or episode count hits a threshold, an alert fires and the website blocker activates.

---

## Features

- **Real-time gaze tracking** with 15-second per-session calibration
- **Multi-signal phone detection** — object detection, hand grip, and body posture work together
- **Focus state machine** with configurable sensitivity (Low / Medium / High)
- **Website blocker** via OS hosts file — works across all browsers, no proxy needed
- **Live dashboard** — focus score, away time, distraction events, session timer
- **Configurable alerts** — time limit, episode count, instant phone alert

---

## System Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.9 – 3.11 (3.10 recommended) |
| RAM | 4 GB minimum |
| OS | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| Webcam | Any USB or built-in camera |
| Browser | Chrome 110+, Edge 110+ |
| GPU | Not required — CPU-only works |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/FocusEye.git
cd FocusEye
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv focuseye-env
focuseye-env\Scripts\activate

# macOS / Linux
python -m venv focuseye-env
source focuseye-env/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the server**
```bash
python app.py
```

On first run, four model files (~30 MB total) are auto-downloaded:

| Model | Size | Purpose |
|-------|------|---------|
| `face_landmarker.task` | ~4 MB | Face mesh + iris |
| `hand_landmarker.task` | ~8 MB | Hand skeleton |
| `pose_landmarker_lite.task` | ~5 MB | Body pose |
| `yolov8n.pt` | ~6.5 MB | Phone object detection |

**5. Open your browser**
```
http://localhost:5000
```

Allow camera access when prompted, then click **START SESSION**.

> **Website blocker requires admin privileges.**
> Windows: run terminal as Administrator.
> macOS / Linux: `sudo python app.py`

---

## Project Structure

```
FocusEye/
├── app.py                     # Flask server — camera loop, REST API, WebSocket events
├── requirements.txt
├── SETUP.md                   # Detailed setup & troubleshooting guide
│
├── ml/
│   ├── focus_engine.py        # ML orchestrator — runs all detectors, draws overlays
│   ├── gaze_tracker.py        # Eye gaze + head pose (MediaPipe FaceMesh + solvePnP)
│   ├── phone_detector.py      # YOLOv8 phone detection (COCO class 67)
│   └── hand_pose_detector.py  # Hand grip + body posture (MediaPipe Hands + Pose)
│
├── focus/
│   ├── tracker.py             # Focus state machine — distraction episodes, alerts
│   └── blocker.py             # OS hosts file website blocker
│
└── templates/
    └── index.html             # Frontend — Socket.IO dashboard + video stream
```

---

## Architecture

All AI/ML runs server-side in Python. The browser receives annotated video frames and metrics over WebSocket at ~30 FPS.

```
Webcam (OpenCV, 1280×720)
        │
        ▼
FocusEngine.analyze(frame)
  ├── GazeTracker        →  looking_away, gaze_ratio, head_yaw, head_pitch
  ├── PhoneDetector      →  phone_detected, bounding_boxes, confidence
  └── HandPoseDetector   →  grip_detected, posture_phone, wrist_raised
        │
        ▼
FocusTracker.update(analysis)   ← state machine
  → is_distracted, focus_score, away_events, alert_needed
        │
        ▼
FocusEngine.draw_overlay(frame, state)
  → annotated JPEG (face mesh, head axes, hand skeleton, phone bbox)
        │
        ▼
SocketIO emit('frame', { image: base64, data: json })
        │
        ▼
Browser (index.html)
  ├── Render frame in <img>
  ├── Update dashboard metrics
  └── Trigger alert modal + sound
```

---

## Distraction Detection Logic

### Gaze Tracking
FaceMesh provides 478 landmarks (468 face + 10 iris per eye). The gaze ratio is the horizontal position of the iris within the eye socket, normalized 0–1. A value near 0.5 means looking at the screen; values below 0.28 or above 0.72 indicate looking away. The first 15 seconds of each session calibrate the user's natural baseline. A 6-frame moving average filters micro-glances.

### Head Pose
Six facial anchor points are matched to a 3D face model via `cv2.solvePnP`. The resulting rotation matrix is decomposed into yaw and pitch angles. Deviations exceeding ±20° yaw or ±22° pitch from the calibrated baseline trigger "looking away."

### Phone Detection (three signals)
1. **YOLOv8-nano** detects COCO class 67 (cell phone) with ≥50% confidence, running every 5 frames to save CPU.
2. **Hand grip detection** checks if 3+ fingers are curled in a phone-holding pose using MediaPipe hand landmarks.
3. **Body posture detection** checks if the head is pitched >90° forward from the baseline (phone-reading posture) with a raised wrist.

### Focus State Machine

```
Every frame:
  currently_away = (looking_away OR phone_detected)

  If currently_away:
    away_frames++
    If away_frames >= 20 (~0.67s):
      is_distracted = True
      away_events++

  If not currently_away AND screen-facing:
    focused_frames++
    If focused_frames >= 15 (~0.5s):
      is_distracted = False

  Alert fires when ANY of:
    • total_away_time >= 2 min (configurable)
    • away_events >= 30
    • phone held for 3+ seconds (instant alert mode)
    (minimum 20-second cooldown between alerts)

  focus_score = focused_time / total_session_time × 100
```

---

## Configuration

Settings are adjustable from the dashboard without restarting the server.

| Setting | Default | Description |
|---------|---------|-------------|
| Away time limit | 2 minutes | Total distracted time before alert |
| Sensitivity | Medium | Low (30 frames) / Medium (20) / High (10) |
| Phone instant alert | Off | Alert immediately on 3s of phone use |
| Blocked sites | See below | Domains redirected to 127.0.0.1 |

**Default blocked sites:** YouTube, Instagram, Twitter/X, TikTok, Reddit, Netflix, Facebook

Sites can be added or removed from the dashboard's website blocker panel at any time.

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sites` | List blocked sites |
| POST | `/api/sites` | Add a site |
| DELETE | `/api/sites` | Remove a site |
| POST | `/api/blocker/toggle` | Enable / disable blocking |
| GET/POST | `/api/settings` | Read / update focus settings |
| POST | `/api/reset` | Reset session state |

**WebSocket events (Socket.IO):**
- `start_session` — start the camera loop
- `stop_session` — stop camera and ML processing
- `frame` — server emits annotated frame + metrics every ~33ms

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not found | Change `cv2.VideoCapture(0)` to `(1)` in `app.py` |
| MediaPipe install fails | Use Python 3.9–3.11; try `pip install mediapipe --pre` |
| Low FPS (<10) | Reduce resolution to 640×480; increase YOLO throttle from 5 to 10 frames |
| Website blocker: Permission denied | Run as Administrator (Windows) or with `sudo` (macOS/Linux) |
| Port 5000 conflict | Change `port=5000` in `app.py` — macOS AirPlay uses 5000, try 5001 |
| YOLO slow or crashes | Comment out `ultralytics` in `requirements.txt` — phone detection is optional |

See `SETUP.md` for detailed platform-specific instructions.

---

## Dependencies

```
Flask==3.0.3
flask-socketio==5.3.6
simple-websocket==1.0.0
opencv-python==4.10.0.84
mediapipe==0.10.33
numpy>=1.24.0,<2.0
ultralytics>=8.0.0
```
