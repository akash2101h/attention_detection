"""
core/data_saver.py – Automatic labelled frame collection.
Saves annotated frames + a CSV log for building a custom dataset.
"""

import cv2
import csv
import os
import time
import threading
import config as C


class DataSaver:
    def __init__(self):
        if not C.SAVE_DATA:
            return
        os.makedirs(C.SAVE_DIR, exist_ok=True)
        self._csv_path = os.path.join(C.SAVE_DIR, "labels.csv")
        self._lock     = threading.Lock()

        # Write CSV header if file is new
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "frame_file",
                    "student_id", "label", "score",
                    "teacher_active",
                ])

    def save(
        self,
        frame: "np.ndarray",
        frame_no: int,
        results: list[dict],
        teacher_active: bool,
    ):
        """Save annotated frame + append rows to CSV."""
        if not C.SAVE_DATA:
            return
        ts   = time.strftime("%Y%m%d_%H%M%S")
        name = f"frame_{ts}_{frame_no:06d}.jpg"
        path = os.path.join(C.SAVE_DIR, name)

        def _write():
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                with open(self._csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    for r in results:
                        w.writerow([
                            ts, name,
                            r["id"], r["label"], r["score"],
                            int(teacher_active),
                        ])

        threading.Thread(target=_write, daemon=True).start()
