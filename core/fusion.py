"""
core/fusion.py – Multi-camera context fusion.

Back camera role
────────────────
The back camera faces the teacher / board.
We use it ONLY for context — not for tracking students.

What we extract:
  • teacher_active : bool  – is someone present near the board?
  • board_activity : float – how much visual change is at the board?
                             (high = teacher writing/moving = class active)

This context is passed to the attention classifier:
  - When teacher is active → students SHOULD be watching → stricter scoring
  - When teacher is absent → students may be on break → lenient scoring
"""

import cv2
import numpy as np
import time


class BackCameraAnalyzer:
    """
    Lightweight analyzer for the back (teacher-facing) camera.
    Uses frame differencing — no YOLO needed on this camera.
    """

    def __init__(self):
        self.prev_gray      : np.ndarray | None = None
        self.teacher_active : bool  = False
        self.activity_score : float = 0.0
        self._last_update   : float = 0.0

        # Simple smoothing
        self._active_smooth : float = 0.0

    def update(self, frame: np.ndarray) -> dict:
        """
        Analyse one back-camera frame.
        Returns context dict used by the attention engine.
        """
        now  = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)

        activity = 0.0
        if self.prev_gray is not None:
            diff    = cv2.absdiff(gray, self.prev_gray)
            _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            # Normalise by frame area
            activity = float(np.sum(mask)) / float(mask.size) * 100.0

        self.prev_gray = gray

        # Smooth activity score
        self._active_smooth = 0.8 * self._active_smooth + 0.2 * activity

        # Teacher considered "active" if smoothed activity > 0.4 %
        self.teacher_active  = self._active_smooth > 0.4
        self.activity_score  = round(self._active_smooth, 2)
        self._last_update    = now

        return self.context()

    def context(self) -> dict:
        return {
            "teacher_active":  self.teacher_active,
            "activity_score":  self.activity_score,
        }
