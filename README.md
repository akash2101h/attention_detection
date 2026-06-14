# Class Attention Detection System

> Real-time student attention monitoring using computer vision and deep learning - deployed as a browser-accessible Flask web application.

---

## What It Does

Teachers in a classroom cannot monitor all 40–60 students simultaneously. This system solves that problem by automatically detecting every student in a classroom camera feed, assigning each one a unique ID, and classifying them as **Attentive**, **Inattentive**, or **Unknown** in real time.

Results stream live to a web dashboard that any teacher can open from any browser on the same network, no software installation required.

---

## Demo

> Dashboard showing live detection on classroom video

<img width="898" height="540" alt="frame_20260426_110759_000300" src="https://github.com/user-attachments/assets/d5a0bbd8-664d-48eb-8466-5f00746a6293" />


> Bounding boxes: 🟢 Green = Attentive | 🟠 Orange = Inattentive | ⬜ Gray = Unknown

---

## Key Features

- **Real-time YOLO11n detection** on NVIDIA GPU via CUDA
- **Custom centroid tracker** built from scratch stable student IDs without DeepSORT
- **Novel head-down detection** using bounding box top-edge (y1) vertical drift no face model needed
- **Grace period mechanism** - prevents false labels from momentary movements
- **0–100 penalty-based attention score** - more informative than binary labels
- **Dual camera support** - back camera detects teacher activity and modulates scoring
- **Adaptive preprocessing** - auto corrects low light and overexposed frames
- **Evidence frame logger** - auto saves annotated frames for post-class review
- **Live web dashboard** - MJPEG stream, live stats, source switching, video upload
- **Active learning ready** - saved frames can be reviewed and used for retraining

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11.9 |
| Deep Learning | PyTorch 2.6.0 + CUDA 12.4 |
| Object Detection | Ultralytics YOLO11n 8.4.39 |
| Video Processing | OpenCV 4.x |
| Web Server | Flask 3.x |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Training Platform | Roboflow |
| GPU | NVIDIA GeForce RTX 3050 (4GB VRAM) |

---

## Project Structure

```
attention/
├── app.py                  ← Flask application entry point
├── pipeline.py             ← Camera processing pipeline (threaded)
├── config.py               ← All configuration values (single source of truth)
├── requirements.txt        ← Python dependencies
├── saved_data/             ← Auto-saved annotated evidence frames
├── uploads/                ← Uploaded video files from dashboard
├── templates/
│   └── index.html          ← Web dashboard UI
└── core/
    ├── __init__.py
    ├── detector.py         ← YOLO11n inference on GPU
    ├── tracker.py          ← Custom centroid multi-object tracker
    ├── attention.py        ← Attention classification engine
    ├── preprocess.py       ← Adaptive CLAHE + gamma preprocessing
    ├── fusion.py           ← Dual camera teacher context fusion
    ├── drawing.py          ← Bounding box and HUD annotation
    └── data_saver.py       ← Evidence frame logger
```

---

## How It Works

```
Camera / Video File
        ↓
Adaptive Preprocessing (CLAHE + Gamma)
        ↓
YOLO11n Person Detection (GPU)
        ↓
Centroid Tracker → Stable Student IDs
        ↓
Attention Engine (y1 drift + movement + grace period)
        ↓
Back Camera Fusion (teacher context)
        ↓
Annotated Frame → Flask MJPEG Stream → Browser Dashboard
        ↓
Evidence Frame Auto-Saver
```

---

## Attention Classification Rules

Each student goes through the following decision tree every frame:

| Priority | Rule | Label |
|---|---|---|
| 1 | Student just appeared (grace period active) | Unknown |
| 2 | Head down confirmed for N seconds | Inattentive |
| 3 | Too many large sudden movements | Inattentive |
| 4 | Too much total movement | Inattentive |
| 5 | YOLO behavior class = inattentive | Inattentive |
| 6 | None of the above | Attentive |

**Attention Score Formula:**
```
Score = 100
      - (2 × small_moves)
      - (6 × large_moves)
      - min(30, head_down_seconds × 5)
      + teacher_active_bonus
      clamp(0, 100)
```

---

## Model Training

| Parameter | Value |
|---|---|
| Base model | YOLO11n pretrained on COCO |
| Dataset | Custom classroom dataset |
| Dataset split | 85% Train / 15% Test |
| Classes | Attentive, Inattentive, Phone |
| mAP@50 | **84.2%** |
| Precision | 79.5% |
| Recall | 77.4% |
| F1 Score | 78.4% |
| Training device | NVIDIA RTX 3050 (CUDA) |

> Previous model trained on Roboflow dataset achieved 70.8% mAP. Custom classroom dataset improved this to **84.2% - a 13.4% gain** through domain-specific training.

---

## Installation

**Step 1 — Clone the repository**
```bash
git clone https://github.com/akash2101h/attention_detection.git
cd attention
```

**Step 2 — Install dependencies**
```bash
py -3.11 -m pip install -r requirements.txt
```

**Step 3 — Verify GPU is available**
```bash
py -3.11 -c "import torch; print(torch.cuda.is_available())"
```
Should print `True`.

**Step 4 — Run the application**
```bash
py -3.11 app.py
```

**Step 5 — Open browser**
```
http://localhost:5000
```

---

## Configuration

All settings are in `config.py`. Change here only - no other file needs editing.

```python
CAMERA_FRONT    = 0                    # 0 = webcam, or path to video file
CAMERA_BACK     = 1                    # back camera index
YOLO_MODEL      = "best.pt"            # model file path
CONF_THRESH     = 0.35                 # detection confidence threshold
FRAME_W         = 640                  # frame width
FRAME_H         = 480                  # frame height
PROCESS_EVERY_N = 2                    # run YOLO every N frames
SAVE_INTERVAL   = 150                  # save frame every N frames
DEVICE          = "cuda"               # cuda or cpu
GAMMA           = 1.8                  # gamma correction value
GRACE_SECONDS   = 3.0                  # grace period before labelling
HEAD_DOWN_Y1_DRIFT    = 20             # pixels of drift to trigger head-down
HEAD_DOWN_CONFIRM_S   = 2.0            # seconds head must stay down
```

---

## Dashboard Features

| Feature | Description |
|---|---|
| Live video stream | MJPEG front and back camera feeds |
| Live statistics | Total students, attentive, inattentive, FPS |
| Attention percentage | Real-time class attention bar |
| Teacher context | Back camera activity status |
| Attention history | Rolling chart of attentive vs inattentive |
| Webcam button | One click switch to laptop webcam |
| Camera 1 button | One click switch to external camera |
| Upload video | Drag and drop or browse video file |
| Advanced source | Enter custom path or RTSP URL |
| Saved frames gallery | Browse auto-saved evidence frames |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel Core i5 | Intel Core i7 |
| RAM | 8 GB | 16 GB |
| GPU | NVIDIA 4GB VRAM + CUDA | NVIDIA RTX 3050+ |
| Storage | 10 GB free | 20 GB free |
| Camera | Any webcam | 1080p webcam or IP camera |

---

## Known Limitations

- Custom model may struggle with heavy student occlusion
- Cannot distinguish academic phone use from social media distraction
- Centroid tracker can occasionally swap IDs when students cross paths
- Not compliant with India's DPDP Act 2023 for institutional deployment
- No automatic session summary report generation after class

---

## Future Work

- [ ] Collect larger annotated classroom dataset for improved accuracy
- [ ] YOLO11-pose keypoint detection for more accurate head-down detection
- [ ] Automatic session summary PDF/CSV report after each class
- [ ] Deploy on fixed IP ceiling camera for full room coverage
- [ ] DPDP Act 2023 compliance for institutional deployment
- [ ] Multi-classroom support via microservices architecture
- [ ] Integration with college LMS for attention-performance correlation
- [ ] Active learning loop - auto-select uncertain frames for retraining

---

## Contributions

| Contribution | Description |
|---|---|
| Bounding box y1 drift | Novel head-down detection - no face model needed |
| Grace period mechanism | Reduces false positive labels from momentary movements |
| Custom centroid tracker | Built from scratch, no DeepSORT dependency |
| Penalty-based 0–100 score | Continuous attention score vs binary label |
| Adaptive preprocessing | Auto-corrects lighting per frame |
| Active learning pipeline | System collects its own retraining data |
| Random classroom dataset | First documented attempt at domain-specific classroom training |

---

** Please 
