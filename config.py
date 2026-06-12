"""
config.py – Central configuration for the Attention Monitoring System.
Edit values here; no other file needs to change.
"""

# ── Video sources ────────────────────────────────────────────
# Use integer (0,1,2) for webcam, string path for video file,
# or RTSP URL for capture card / DVR feed.
CAMERA_FRONT = r"C:\Users\agk21\OneDrive\Desktop\classroom\test.mp4"          
CAMERA_BACK  = 1          # faces teacher / board
                          # e.g. "rtsp://192.168.1.100:554/stream1"
                          # e.g. "classroom_front.mp4"

# ── Frame settings ───────────────────────────────────────────
FRAME_W = 960
FRAME_H = 540
PROCESS_EVERY_N = 2       # run YOLO every N frames (1 = every frame)

# ── YOLO ─────────────────────────────────────────────────────
YOLO_MODEL = "yolo11n.pt"   
DEVICE = "cuda"
CONF_THRESH  = 0.35
NMS_IOU      = 0.45
MIN_BOX_AREA = 800             # px² – ignore tiny detections

# ── Tracking ─────────────────────────────────────────────────
MAX_DISAPPEARED = 40
MAX_DIST        = 90           # px – max centroid jump per frame

# ── Attention analysis window ────────────────────────────────
WINDOW_SECONDS  = 8            # sliding window length
FPS_ESTIMATE    = 15           # used to size the deque
GRACE_SECONDS   = 4            # seconds before any label is given
                               # (student gets benefit of the doubt)

# ── Movement thresholds ──────────────────────────────────────
MOVE_THRESH_PX       = 8       # centroid shift > this = "moving"
LARGE_MOVE_THRESH    = 28      # centroid shift > this = "large move"

# ── Head-down detection ──────────────────────────────────────
# We track the TOP EDGE of the bounding box (y1).
# If y1 creeps downward significantly the person's head has
# dropped forward (head-down / bent over desk).
HEAD_DOWN_Y1_DRIFT   = 18      # px – how much y1 must drop vs its baseline
HEAD_DOWN_CONFIRM_S  = 3.0     # seconds head must stay down before flagging

# ── Back-camera context ──────────────────────────────────────
# When the back camera detects a person near the board/front,
# we assume the teacher is active → raise attention bar slightly.
TEACHER_ACTIVE_BOOST = 8       # score bonus when teacher is at board

# ── Attention scoring (absolute, not percentage thresholds) ──
# Scores are built from accumulated event counts, not ratios.
HIGH_MOVE_COUNT      = 12      # moves in window → inattentive
LARGE_MOVE_COUNT     = 3       # large moves in window → inattentive

# ── Low-light ────────────────────────────────────────────────
GAMMA = 1.8

# ── Flask / streaming ────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
JPEG_QUALITY = 75              # MJPEG stream quality (1-95)

# ── Data collection ──────────────────────────────────────────
SAVE_DATA      = True
SAVE_DIR       = "saved_data"
SAVE_INTERVAL  = 150            # save a labelled frame every N frames
