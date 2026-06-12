import os
import time
import glob
import json
import threading

from flask import (
    Flask, Response, render_template,
    jsonify, send_from_directory, request,
)

import config as C
from core.fusion     import BackCameraAnalyzer
from pipeline        import CameraProcessor, BackCameraProcessor

# ── Initialise processors ─────────────────────────────────────
back_analyzer  = BackCameraAnalyzer()
front_proc     = CameraProcessor(C.CAMERA_FRONT, back_analyzer)
back_proc      = BackCameraProcessor(C.CAMERA_BACK, back_analyzer)

app = Flask(__name__)

# ── Start camera threads on first request ─────────────────────
_started = False
_lock    = threading.Lock()

def _ensure_started():
    global _started
    with _lock:
        if not _started:
            front_proc.start()
            back_proc.start()
            _started = True


# ══════════════════════════════════════════════
#  MJPEG stream generators
# ══════════════════════════════════════════════

def _gen_stream(processor):
    """Yield MJPEG frames from a processor."""
    _ensure_started()
    placeholder = _make_placeholder("No signal")
    while True:
        frame = processor.latest_jpeg() or placeholder
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )
        time.sleep(1 / 30)   # cap at 30 fps to browser


def _make_placeholder(text: str) -> bytes:
    """Generate a simple grey JPEG placeholder."""
    import cv2, numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, text, (220, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 160, 160), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ══════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════

@app.route("/")
def index():
    _ensure_started()
    return render_template("index.html")


@app.route("/stream/front")
def stream_front():
    return Response(
        _gen_stream(front_proc),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/back")
def stream_back():
    return Response(
        _gen_stream(back_proc),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    """Live statistics polled by the dashboard every second."""
    s = front_proc.stats()
    return jsonify({
        "total":          s.get("total", 0),
        "attentive":      s.get("attentive", 0),
        "inattentive":    s.get("inattentive", 0),
        "teacher_active": s.get("teacher_active", False),
        "fps":            s.get("fps", 0.0),
        "attention_pct":  _pct(s.get("attentive", 0), s.get("total", 0)),
    })


@app.route("/api/recordings")
def api_recordings():
    """List saved video files for playback."""
    pattern = os.path.join(C.SAVE_DIR, "*.jpg")
    files   = sorted(glob.glob(pattern), reverse=True)[:50]
    return jsonify([os.path.basename(f) for f in files])


@app.route("/saved_data/<path:filename>")
def saved_file(filename):
    return send_from_directory(C.SAVE_DIR, filename)


@app.route("/api/source", methods=["POST"])
def update_source():
    """
    Hot-swap camera source without restarting the server.
    POST JSON: {"front": "0", "back": "1"}
    """
    data = request.get_json(force=True)
    if "front" in data:
        src = data["front"]
        C.CAMERA_FRONT = int(src) if str(src).isdigit() else src
        front_proc.stop()
        front_proc.source = C.CAMERA_FRONT
        front_proc.start()
    if "back" in data:
        src = data["back"]
        C.CAMERA_BACK = int(src) if str(src).isdigit() else src
        back_proc.stop()
        back_proc.source = C.CAMERA_BACK
        back_proc.start()
    return jsonify({"status": "ok"})


# ── helpers ───────────────────────────────────────────────────

def _pct(part, total):
    return round(100 * part / total) if total else 0


# ══════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[Flask] Starting on http://{C.FLASK_HOST}:{C.FLASK_PORT}")
    app.run(
        host=C.FLASK_HOST,
        port=C.FLASK_PORT,
        threaded=True,
        debug=False,
    )
