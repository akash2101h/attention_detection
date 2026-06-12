"""
pipeline.py – Main camera processing engine.
Runs both cameras in separate threads.
Exposes latest JPEG frames + statistics for Flask to serve.
"""

import cv2
import time
import threading
import numpy as np
from collections import defaultdict

import config as C
from core.preprocess import enhance
from core.detector   import PersonDetector
from core.tracker    import CentroidTracker
from core.attention  import StudentState, classify
from core.fusion     import BackCameraAnalyzer
from core.drawing    import draw_student, draw_hud, draw_back_camera
from core.data_saver import DataSaver


class CameraProcessor:
    """
    Processes one camera stream (front camera — student facing).
    Detection + tracking + attention analysis all happen here.
    """

    def __init__(self, source, back_analyzer: BackCameraAnalyzer):
        self.source       = source
        self.back         = back_analyzer
        self.detector     = PersonDetector()
        self.tracker      = CentroidTracker()
        self.saver        = DataSaver()
        self.states       : dict[int, StudentState] = defaultdict(StudentState)

        self._frame_lock  = threading.Lock()
        self._jpeg        : bytes | None = None
        self._stats       : dict = {
            "total": 0, "attentive": 0, "inattentive": 0,
            "teacher_active": False, "fps": 0.0,
        }
        self._running     = False
        self._thread      : threading.Thread | None = None

    # ── public API ────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._jpeg

    def stats(self) -> dict:
        with self._frame_lock:
            return dict(self._stats)

    # ── main loop ─────────────────────────────────────────────

    def _loop(self):
        cap = self._open_source()
        if cap is None:
            return

        frame_no  = 0
        fps       = 0.0
        t_prev    = time.time()
        window_f  = int(C.WINDOW_SECONDS * C.FPS_ESTIMATE)

        while self._running:
            ret, raw = cap.read()
            if not ret:
                # Loop video files
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_no += 1
            now    = time.time()
            fps    = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            # Resize + enhance
            frame    = cv2.resize(raw, (C.FRAME_W, C.FRAME_H))
            enhanced = enhance(frame.copy())

            # Get back-camera context
            ctx     = self.back.context()
            teacher = ctx["teacher_active"]

            # Detection (every N frames)
            if frame_no % C.PROCESS_EVERY_N == 0:
                dets = self.detector.detect(enhanced)
            else:
                dets = []

            # Tracking
            tracked = self.tracker.update(dets)

            # Attention analysis
            results = []
            counts  = defaultdict(int)

            for oid, info in tracked.items():
                centroid = info["centroid"]
                bbox     = info["bbox"]

                state = self.states[oid]
                state.observe(centroid, bbox)

                label, score = classify(state, teacher)
                counts[label] += 1
                head_down = state.head_down_confirmed()

                draw_student(frame, oid, bbox, label, score, centroid, head_down)
                results.append({
                    "id": oid, "label": label, "score": score,
                    "head_down": head_down,
                })

            # Clean up lost tracks
            for oid in list(self.states):
                if oid not in tracked:
                    del self.states[oid]

            # HUD overlay
            total      = len(tracked)
            attentive  = counts.get("Attentive",   0)
            inattentive= counts.get("Inattentive", 0)
            draw_hud(
                frame, fps, total, attentive, inattentive,
                teacher, "FRONT CAM", frame_no,
            )

            # Data saving
            if C.SAVE_DATA and frame_no % C.SAVE_INTERVAL == 0 and results:
                self.saver.save(frame, frame_no, results, teacher)

            # Encode to JPEG for streaming
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_QUALITY],
            )
            with self._frame_lock:
                if ok:
                    self._jpeg = buf.tobytes()
                self._stats = {
                    "total":          total,
                    "attentive":      attentive,
                    "inattentive":    inattentive,
                    "teacher_active": teacher,
                    "fps":            round(fps, 1),
                }

        cap.release()

    def _open_source(self) -> cv2.VideoCapture | None:
        src = self.source
        # Convert "0" / "1" strings to int for webcam
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, int):
    	    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
    	    cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
        if not cap.isOpened():
            print(f"[Pipeline] ERROR: Cannot open source {src!r}")
            return None
        print(f"[Pipeline] Front camera opened: {src!r}")
        return cap


class BackCameraProcessor:
    """
    Lightweight processor for the back (teacher-facing) camera.
    Only does frame differencing — no YOLO.
    """

    def __init__(self, source, analyzer: BackCameraAnalyzer):
        self.source   = source
        self.analyzer = analyzer
        self._lock    = threading.Lock()
        self._jpeg    : bytes | None = None
        self._running = False
        self._thread  : threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _loop(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
        if not cap.isOpened():
            print(f"[Pipeline] WARNING: Back camera {src!r} not available.")
            return
        print(f"[Pipeline] Back camera opened: {src!r}")

        while self._running:
            ret, raw = cap.read()
            if not ret:
                if isinstance(self.source, str) and not str(self.source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame = cv2.resize(raw, (C.FRAME_W, C.FRAME_H))
            ctx   = self.analyzer.update(frame)
            draw_back_camera(frame, ctx["activity_score"], ctx["teacher_active"])

            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_QUALITY],
            )
            with self._lock:
                if ok:
                    self._jpeg = buf.tobytes()

        cap.release()
