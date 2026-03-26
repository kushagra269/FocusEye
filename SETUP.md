# FocusEye — Setup & Requirements Guide

## What this app does
FocusEye is an AI-powered procrastination blocker that uses your webcam to:
- Track where your eyes are looking (iris gaze tracking)
- Estimate your head direction (head pose via solvePnP)
- Detect if you pick up a smartphone (YOLOv8 object detection)
- Alert you with sound + popup after 2 minutes of distraction
- Optionally block distracting websites via the OS hosts file

All ML runs in Python (server-side). The browser just receives the annotated video stream.

---

## System Requirements

| Item | Minimum |
|------|---------|
| Python | 3.9 – 3.11 (3.10 recommended) |
| RAM | 4 GB |
| CPU | Any modern x86-64 or Apple Silicon |
| OS | Windows 10/11 · macOS 12+ · Ubuntu 20.04+ |
| Webcam | Any USB or built-in camera |
| Browser | Chrome 110+ or Edge 110+ |

> GPU is **not required**. MediaPipe and YOLOv8-nano run efficiently on CPU.

---

## Step-by-Step Installation

### 1. Install Python 3.10
Download from https://python.org and check **"Add to PATH"** during install.

Verify:
```
python --version
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv focuseye-env

# Windows
focuseye-env\Scripts\activate

# macOS / Linux
source focuseye-env/bin/activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **Flask + Flask-SocketIO** — web server & real-time WebSocket
- **OpenCV** — camera capture, frame encoding
- **MediaPipe** — face mesh (468 landmarks) + iris tracking (10 points) + head pose
- **ultralytics / YOLOv8-nano** — phone detection (~6 MB model download on first run)
- **NumPy 1.26** — required by MediaPipe (incompatible with NumPy 2.x)

### 4. Run the server
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```
Press **► START SESSION** and allow camera access.

---

## How the AI/ML works

### Gaze Tracking (iris detection)
MediaPipe FaceMesh with `refine_landmarks=True` returns **478 facial landmarks**:
- 468 face landmarks (standard mesh)
- 10 iris landmarks (5 per eye: center + 4 edges)

**Gaze ratio** = horizontal position of iris center within the eye socket (0–1).
- ~0.5 = looking at screen
- < 0.28 = looking left
- > 0.72 = looking right

### Head Pose Estimation (solvePnP)
6 facial anchor points (nose tip, chin, eye corners, cheeks) are matched to a
3D generic face model using `cv2.solvePnP`. The resulting rotation matrix is
decomposed with `cv2.RQDecomp3x3` to get:
- **Yaw** — horizontal head rotation (left/right)
- **Pitch** — vertical head rotation (up/down)

Thresholds: yaw > ±20°, pitch > ±22° triggers "looking away".

### Phone Detection (YOLOv8-nano)
YOLOv8-nano is a 6 MB YOLO model trained on COCO dataset.
COCO class 67 = "cell phone". Runs every 5 frames to save CPU.
First launch downloads `yolov8n.pt` automatically from ultralytics CDN.

### Focus State Machine
```
Frame arrives
   ↓
currently_away = (looking_away OR phone_detected)
   ↓
away_frames counter increments / resets
   ↓
After 20 consecutive away frames (~0.67s):
  → is_distracted = True
  → distraction timer starts
  → away_events += 1
   ↓
After 15 consecutive focused frames (~0.5s):
  → accumulate away time
  → is_distracted = False
   ↓
Alert fires when:
  total_away_time >= limit (default 2 min)
  OR  away_events >= 30
  OR  phone held for 3+ seconds (instant alert mode)
```

---

## Website Blocker

The website blocker modifies the OS **hosts file** to redirect blocked domains
to `127.0.0.1` (localhost), making them unreachable in any browser.

### Windows — requires Administrator
Right-click the terminal → **Run as administrator**, then:
```bash
python app.py
```

### macOS / Linux — requires sudo
```bash
sudo python app.py
```

### Hosts file locations
| OS | Path |
|----|------|
| Windows | `C:\Windows\System32\drivers\etc\hosts` |
| macOS / Linux | `/etc/hosts` |

> If you don't have admin rights, the blocker toggle will show a permission error.
> The rest of the app (gaze tracking, alerts) still works without admin rights.

---

## Project Structure

```
FocusEye/
├── app.py                  ← Flask + SocketIO server (entry point)
├── requirements.txt        ← Python dependencies
├── SETUP.md                ← This file
│
├── ml/
│   ├── gaze_tracker.py     ← MediaPipe FaceMesh, iris gaze, solvePnP head pose
│   ├── phone_detector.py   ← YOLOv8 phone detection
│   └── focus_engine.py     ← Orchestrates ML + draws OpenCV annotations
│
├── focus/
│   ├── tracker.py          ← Focus state machine, score, alert logic
│   └── blocker.py          ← OS hosts file website blocker
│
└── templates/
    └── index.html          ← Complete frontend (Socket.IO + dashboard UI)
```

## Data Flow

```
Webcam (OpenCV)
    ↓
focus_engine.analyze()
    ├── GazeTracker.process()    → gaze ratio, yaw, pitch, looking_away
    └── PhoneDetector.detect()   → list of phone bounding boxes
    ↓  (annotated JPEG frame)
FocusTracker.update()
    → is_distracted, total_away_ms, focus_score, alert_needed
    ↓
Flask-SocketIO.emit('frame', { image: base64_jpeg, data: json })
    ↓
Browser (JavaScript)
    → renders frame in <img>
    → updates dashboard (score ring, metrics, progress bars)
    → triggers alert modal + sound if alert_needed
```

---

## Troubleshooting

**Camera not found**
- Make sure no other app is using the webcam.
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py`.

**mediapipe install fails**
- Requires Python 3.9–3.11. Python 3.12 is not yet supported by mediapipe 0.10.x.
- Try: `pip install mediapipe --pre` for a prerelease build.

**ultralytics / YOLO slow or fails**
- Phone detection is optional. The app works without it.
- Comment out `ultralytics` in `requirements.txt` if you don't need it.

**Website blocker: Permission denied**
- Run the terminal as Administrator (Windows) or with `sudo` (Mac/Linux).

**Low FPS (< 10)**
- Reduce camera resolution in `app.py`: change `1280 × 720` to `640 × 480`.
- YOLO is the slowest part. Increase `_PHONE_INTERVAL` in `focus_engine.py` from 5 to 10.

**Port 5000 already in use**
- Change `port=5000` to another port in `app.py`.
- On macOS, port 5000 is used by AirPlay. Use port 5001.
