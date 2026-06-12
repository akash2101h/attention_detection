"""
core/drawing.py – All OpenCV drawing helpers.
Keeps visual logic out of the main pipeline.
"""

import cv2
import numpy as np

COLORS = {
    "Attentive":   (0, 200, 80),
    "Inattentive": (0, 100, 240),
    "Unknown":     (160, 160, 160),
}


def draw_student(
    frame: np.ndarray,
    obj_id: int,
    bbox: tuple,
    label: str,
    score: int,
    centroid: tuple,
    head_down: bool = False,
) -> None:
    x1, y1, x2, y2 = bbox
    color = COLORS.get(label, COLORS["Unknown"])

    # Bounding box
    thickness = 2 if label != "Inattentive" else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Score bar (thin bar just above the box)
    bar_w  = x2 - x1
    filled = int(bar_w * score / 100)
    cv2.rectangle(frame, (x1, y1 - 7), (x1 + filled, y1), color, -1)
    cv2.rectangle(frame, (x1, y1 - 7), (x2,           y1), color,  1)

    # Label pill
    extra = "  [head down]" if head_down else ""
    tag   = f"#{obj_id} {label}{extra}  {score}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    pill_y = max(y1 - 24, 0)
    cv2.rectangle(frame, (x1, pill_y), (x1 + tw + 6, pill_y + th + 6), color, -1)
    cv2.putText(
        frame, tag, (x1 + 3, pill_y + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Centroid dot
    cv2.circle(frame, centroid, 3, color, -1)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    total: int,
    attentive: int,
    inattentive: int,
    teacher_active: bool,
    cam_label: str,
    frame_no: int,
) -> None:
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)

    teacher_str = "Teacher: ACTIVE" if teacher_active else "Teacher: away"
    teacher_col = (0, 220, 100) if teacher_active else (100, 100, 100)

    info = (
        f"{cam_label}  |  FPS:{fps:4.1f}  |  "
        f"Students:{total}  Att:{attentive}  Inatt:{inattentive}  |  "
        f"Frame:{frame_no}"
    )
    cv2.putText(frame, info, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, teacher_str, (w - 180, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, teacher_col, 1, cv2.LINE_AA)


def draw_back_camera(frame: np.ndarray, activity: float, teacher_active: bool) -> None:
    """Minimal overlay for the back camera view."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)
    status = "ACTIVE" if teacher_active else "idle"
    color  = (0, 220, 100) if teacher_active else (100, 100, 100)
    cv2.putText(
        frame,
        f"Back Cam  |  Board activity:{activity:.1f}%  |  Teacher:{status}",
        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA,
    )
