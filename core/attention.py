"""
core/attention.py – Attention analysis engine.

Design philosophy
─────────────────
• NO posture / angle / h-w ratio heuristics  (unreliable with single cam)
• NO percentage thresholds for classification
• Uses ABSOLUTE EVENT COUNTS over a time window
• Head-down detection via bounding-box TOP EDGE drift
• Grace period: student gets benefit of the doubt for first N seconds
• Teacher-active context from back camera boosts attention scores

Labels
──────
  Attentive   – stable position, minimal movement, not head-down
  Inattentive – excess movement OR confirmed head-down for too long

Head-down logic
───────────────
When a student lowers their head (looking at phone/book/desk), the
TOP edge of their bounding box (y1) drifts DOWNWARD in the frame
(y1 increases in OpenCV coordinates).

We track a rolling baseline of y1. If the current y1 is consistently
HIGHER (numerically) than the baseline by HEAD_DOWN_Y1_DRIFT pixels
AND this persists for HEAD_DOWN_CONFIRM_S seconds, we flag head-down.

Exception: if the teacher is NOT active at the board (back camera),
the head-down threshold is relaxed slightly — student might be copying
notes that were just written.
"""

import time
import math
from collections import deque
import config as C

ATTENTIVE_CLASSES  = {0, 3, 8}       
INATTENTIVE_CLASSES = {1, 2, 4, 5, 6, 7, 9}  

class StudentState:
    """
    Per-student sliding window of observations.
    All deques are bounded to WINDOW_SECONDS * FPS_ESTIMATE frames.
    """

    def __init__(self):
        wf = int(C.WINDOW_SECONDS * C.FPS_ESTIMATE)
        self.centroids    : deque = deque(maxlen=wf)
        self.y1_history   : deque = deque(maxlen=wf)   # top-edge of bbox
        self.move_events  : deque = deque(maxlen=wf)   # bool
        self.large_events : deque = deque(maxlen=wf)   # bool
        self.timestamps   : deque = deque(maxlen=wf)

        self.first_seen    : float = time.time()
        self.head_down_since: float | None = None      # timestamp head went down
        self.y1_baseline   : float | None = None       # rolling mean of y1 when upright
        self.yolo_class : int | None = None
    # ── called every tracked frame ────────────────────────────
    def observe(self, centroid: tuple, bbox: tuple, yolo_class: int | None = None):
        cx, cy         = centroid
        x1, y1, x2, y2 = bbox
        now            = time.time()

        if yolo_class is not None:
            self.yolo_class = yolo_class

        # ── Movement ──────────────────────────────────────────
        moved = large = False
        if self.centroids:
            px, py = self.centroids[-1]
            dist   = math.hypot(cx - px, cy - py)
            moved  = dist > C.MOVE_THRESH_PX
            large  = dist > C.LARGE_MOVE_THRESH

        self.centroids.append((cx, cy))
        self.y1_history.append(y1)
        self.move_events.append(moved)
        self.large_events.append(large)
        self.timestamps.append(now)

        # ── Head-down tracking ────────────────────────────────
        if len(self.y1_history) >= 6:
            recent_y1 = list(self.y1_history)[-6:]
            median_y1 = sorted(recent_y1)[len(recent_y1) // 2]

            if self.y1_baseline is None:
                self.y1_baseline = float(median_y1)
            else:
                drift = median_y1 - self.y1_baseline

                if drift >= C.HEAD_DOWN_Y1_DRIFT:
                    if self.head_down_since is None:
                        self.head_down_since = now
                else:
                    self.head_down_since = None
                    self.y1_baseline = 0.85 * self.y1_baseline + 0.15 * median_y1

    # ── query helpers ─────────────────────────────────────────

    def in_grace_period(self) -> bool:
        return (time.time() - self.first_seen) < C.GRACE_SECONDS

    def head_down_confirmed(self) -> bool:
        """
        True only when head has been down continuously for
        HEAD_DOWN_CONFIRM_S seconds AND student is past grace period.
        """
        if self.head_down_since is None:
            return False
        duration = time.time() - self.head_down_since
        return duration >= C.HEAD_DOWN_CONFIRM_S

    def move_count(self) -> int:
        return sum(self.move_events)

    def large_move_count(self) -> int:
        return sum(self.large_events)

    def enough_data(self) -> bool:
        return len(self.centroids) >= max(6, int(C.FPS_ESTIMATE * 1.5))


# ─────────────────────────────────────────────────────────────
#  Attention classifier
# ─────────────────────────────────────────────────────────────

def classify(state: StudentState, teacher_active: bool) -> tuple[str, int]:
    """
    Classify one student's attention state.

    Returns (label, score) where:
      label : "Attentive" | "Inattentive" | "Unknown"
      score : 0–100  (higher = more attentive)

    Decision logic (NO percentage thresholds — uses raw counts):
    ────────────────────────────────────────────────────────────
    1. Grace period → "Unknown"  (don't judge too quickly)

    2. Head-down confirmed → "Inattentive"
       EXCEPTION: if teacher is NOT active, we give extra leniency
       (student may be copying notes) — requires longer confirmation.

    3. Too many large sudden moves → "Inattentive"
       (standing up, turning around, repeated getting up)

    4. Too many total moves → "Inattentive"
       (constant fidgeting, turning around)

    5. Everything else → "Attentive"

    Score computation (additive penalty system, NOT ratio-based):
    ────────────────────────────────────────────────────────────
    Start at 100.
    Subtract fixed penalties per event count:
      - Each move      → -2 pts
      - Each large move → -6 pts
      - Head-down duration → up to -30 pts
    Add teacher-active bonus.
    Clamp 0–100.
    """
    if not state.enough_data() or state.in_grace_period():
        return "Unknown", 50

    moves  = state.move_count()
    lmoves = state.large_move_count()

    # ── 1. Head-down check ────────────────────────────────────
    # If teacher is not active, require longer confirmation
    # (note-copying allowance)
    effective_confirm = C.HEAD_DOWN_CONFIRM_S
    if not teacher_active:
        effective_confirm = C.HEAD_DOWN_CONFIRM_S + 1.5

    head_down = (
        state.head_down_since is not None
        and (time.time() - state.head_down_since) >= effective_confirm
    )

    # ── 2. Movement check ─────────────────────────────────────
    too_many_large = lmoves >= C.LARGE_MOVE_COUNT
    too_much_move  = moves  >= C.HIGH_MOVE_COUNT

    # ── Classification ────────────────────────────────────────
    yolo_inattentive = getattr(state, 'yolo_class', None) in INATTENTIVE_CLASSES
    if head_down or too_many_large or too_much_move or yolo_inattentive:
        label = "Inattentive"
    else:
        label = "Attentive"

    # ── Score: penalty-based (not ratio-based) ────────────────
    score = 100
    score -= moves  * 2           # small moves
    score -= lmoves * 6           # large moves
    if state.head_down_since is not None:
        down_secs = time.time() - state.head_down_since
        score -= min(30, int(down_secs * 5))   # up to -30 for head-down
    if teacher_active:
        score += C.TEACHER_ACTIVE_BOOST
    score = max(0, min(100, score))

    return label, score
