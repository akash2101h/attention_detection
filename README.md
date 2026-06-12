# Student Attention Detection System
### Real-Time · Rule-Based · No Training Required

---

## 1. Installation

### Step 1 – Python version
Requires **Python 3.9 or higher**.
```
python --version
```

### Step 2 – Install all dependencies
```bash
pip install ultralytics opencv-python numpy scipy
```

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 (pre-trained, no training needed) |
| `opencv-python` | Video capture, image processing, display |
| `numpy` | Array math, frame operations |
| `scipy` | Optional distance utilities |

### Step 3 – YOLO model download
On the **first run**, YOLOv8 nano weights (~6 MB) download automatically.
No extra step needed. To pre-download manually:
```python
from ultralytics import YOLO
YOLO("yolov11n.pt")   # downloads yolov11n.pt to ./
```

---

## 2. How to Run

### Webcam (default)
```bash
python app.py
```

### Video file
```bash
python app.py --source classroom.mp4
```

### All CLI options
```
--source      0 (webcam) or path to video file    default: 0
--model       YOLO weights file                   default: yolov8n.pt
--conf        detection confidence 0.0–1.0        default: 0.40
--width       frame width (px)                    default: 640
--height      frame height (px)                   default: 480
--skip        process every N-th frame            default: 2
--window      attention window in seconds         default: 7
--lowlight    simulate dark classroom
--blur        simulate camera blur
--noise       simulate sensor noise
```

### Keyboard shortcuts (inside the OpenCV window)
| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Save screenshot as `screenshot_<timestamp>.jpg` |

---

## 3. Switching Between Webcam and Video File

**Method A – CLI flag (recommended)**
```bash
# Webcam
python app.py --source 0

# Video file
python app.py --source classroom.mp4
```

**Method B – Edit the config dict in the script**
```python
CFG = {
    "SOURCE": 0,               # ← webcam
    "SOURCE": "classroom.mp4", # ← video file
    ...
}
```

---

## 4. Architecture Overview

```
Raw Frame
   │
   ▼
┌──────────────────┐
│  PRE-PROCESSING  │  CLAHE + Gamma + Bilateral filter
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  YOLO DETECTION  │  Person class (class 0) only
│  (every N frames)│  Filters out tiny detections
└──────────────────┘
   │  list of bboxes + conf
   ▼
┌──────────────────┐
│ CENTROID TRACKER │  Assigns persistent IDs
│                  │  Greedy distance matching
└──────────────────┘
   │  {id: centroid, bbox}
   ▼
┌──────────────────┐
│ STUDENT BUFFER   │  Per-student sliding window
│                  │  Stores: moved, large_move, slumped
└──────────────────┘
   │  feature history
   ▼
┌──────────────────┐
│ ATTENTION LOGIC  │  Rule-based classifier
│  (rule-based)    │  → Attentive / Distracted / Drowsy
└──────────────────┘
   │  label + score (0–100)
   ▼
┌──────────────────┐
│   HUD DISPLAY    │  Bounding box, ID pill, score bar
└──────────────────┘
```

---

## 5. How Tracking Works

The **CentroidTracker** uses centroid matching:

1. Compute the center point (cx, cy) of each detected bounding box.
2. Build a distance matrix between existing tracks and new detections.
3. Match greedily by shortest distance (like a simplified Hungarian algorithm).
4. If distance > `MAX_DIST` (80 px by default), treat as a new student.
5. If a track is unmatched for `MAX_DISAPPEARED` (30) frames, remove it.

**Why not SORT/DeepSORT?**
Centroid tracking is sufficient here because:
- Classroom cameras are mostly static.
- Students don't cross each other rapidly.
- It needs zero extra model weights.

---

## 6. How Attention Logic Works

All decisions are **rule-based** – no ML model is trained.

### Features extracted per frame
| Feature | How it's computed |
|---|---|
| `moved` | Centroid shifted > 12 px from last frame |
| `large_move` | Centroid shifted > 35 px |
| `slumped` | Bounding-box h/w ratio < 1.10 (person appears wider than tall → leaning forward) |

### Classification rules (evaluated top to bottom)

**DROWSY**
- ≥ 50 % of window frames are `slumped`
- AND movement rate < 20 %
- Score: 5–30

**DISTRACTED**
- ≥ 40 % of window frames have `moved`
- OR ≥ 2 `large_move` events in the window
- Score: 15–50

**ATTENTIVE** (everything else)
- Stable, upright posture, low movement
- Score: 70–100

### Why a sliding window (not frame-by-frame)?
- A single sneeze would look "distracted" frame-by-frame.
- A 7-second window smooths transient events.
- `WINDOW_SECONDS` is configurable via `--window`.

---

## 7. Model Variants

| Model | Speed | Accuracy | When to use |
|---|---|---|---|
| `yolov8n.pt` | ★★★★★ | ★★★ | Default – Raspberry Pi / laptops |
| `yolov8s.pt` | ★★★★ | ★★★★ | Good laptop / workstation |
| `yolov8m.pt` | ★★★ | ★★★★★ | GPU workstation / accuracy-first |

Switch with:
```bash
python app.py --model yolov8s.pt
```

---

## 8. Tuning for Your Classroom

| Scenario | Recommended change |
|---|---|
| Many false positives (chairs detected) | Raise `--conf 0.55` |
| Students at the back missed | Lower `--conf 0.30`, set `MIN_BOX_AREA: 800` in CFG |
| Everything labelled "Distracted" | Raise `DISTRACTED_MOVE_RATIO` to `0.55` |
| Drowsy misclassified | Raise `DROWSY_SLUMP_RATIO` to `0.65` |
| Low FPS | Increase `--skip 3` or use `yolov8n.pt` |
| Dark classroom | Use `--lowlight` to verify enhancement; tune `GAMMA` in CFG |

---

## 9. File Structure

```
student_attention_system.py   ← single-file, fully self-contained
README.md                     ← this guide
classroom.mp4                 ← (optional) your test video
yolov8n.pt                    ← auto-downloaded on first run
```

---

## 10. Quick Troubleshooting

**`Cannot open video source: 0`**
→ Your webcam index may differ. Try `--source 1` or `--source 2`.

**Very low FPS**
→ Use `--skip 3` and/or `--model yolov8n.pt`.

**No students detected**
→ Lower confidence: `--conf 0.25`. Check lighting – try `--lowlight` mode to see if enhancement helps.

**`ImportError: No module named ultralytics`**
→ Run `pip install ultralytics` and retry.

## Model Weights

Download the required YOLO weights from Ultralytics:

- yolo11n.pt
- yolov8n.pt