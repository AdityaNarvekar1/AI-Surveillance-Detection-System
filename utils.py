"""
utils.py
--------
Shared helpers: frame pre-processing, timestamp formatting, FPS control,
event logging, and snapshot saving.
"""

import csv
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LOG_DIR      = Path(__file__).parent / "logs"
SNAPSHOT_DIR = LOG_DIR / "snapshots"
CSV_LOG      = LOG_DIR / "detections.csv"
JSON_LOG     = LOG_DIR / "detections.json"

CSV_HEADERS = ["timestamp_video", "timestamp_wall", "event_type", "confidence", "details"]


# ── Directory bootstrap ────────────────────────────────────────────────────────

def ensure_log_dirs():
    LOG_DIR.mkdir(exist_ok=True)
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    if not CSV_LOG.exists():
        with open(CSV_LOG, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


# ── Timestamp helpers ──────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """Convert float seconds → 'HH:MM:SS'."""
    return str(timedelta(seconds=int(seconds)))


def wall_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Video info ─────────────────────────────────────────────────────────────────

def get_video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


# ── Frame pre-processing ───────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray, size=(224, 224)) -> np.ndarray:
    """
    Resize + normalize a BGR frame for the violence model.
    Returns float32 array in [0, 1] with shape (H, W, 3) in RGB order.
    """
    resized    = cv2.resize(frame, size)
    rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized


def build_sequence_input(frame_buffer: list, target_len: int, size=(224, 224)) -> np.ndarray:
    """
    Build a (1, T, H, W, 3) array from the rolling buffer.
    Zero-pads at the start if fewer than target_len frames are available.
    """
    processed = [preprocess_frame(f, size) for f in frame_buffer[-target_len:]]
    while len(processed) < target_len:
        processed.insert(0, np.zeros((*size, 3), dtype=np.float32))
    arr = np.stack(processed, axis=0)    # (T, H, W, 3)
    return np.expand_dims(arr, axis=0)   # (1, T, H, W, 3)


# ── Logging ────────────────────────────────────────────────────────────────────

def log_event(event_type: str, video_ts: str, confidence: float, details: str = ""):
    """Append one detection event to CSV and JSON logs."""
    ensure_log_dirs()
    row = {
        "timestamp_video": video_ts,
        "timestamp_wall":  wall_timestamp(),
        "event_type":      event_type,
        "confidence":      round(confidence, 4),
        "details":         details,
    }
    # CSV
    with open(CSV_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)
    # JSON (one object per line)
    with open(JSON_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")

    logger.info("Event logged: %s", row)


def get_log_path() -> Path:
    return CSV_LOG


# ── Snapshot saving ────────────────────────────────────────────────────────────

def save_snapshot(frame: np.ndarray, video_ts: str, event_type: str) -> str:
    """Save an annotated frame as a JPEG snapshot."""
    ensure_log_dirs()
    safe_ts    = video_ts.replace(":", "-")
    safe_event = event_type.replace(" ", "_")
    filename   = SNAPSHOT_DIR / f"{safe_event}_{safe_ts}.jpg"
    cv2.imwrite(str(filename), frame)
    logger.info("Snapshot saved: %s", filename)
    return str(filename)


# ── FPS control ────────────────────────────────────────────────────────────────

class FPSController:
    """Throttle frame processing to approximate the original video FPS."""

    def __init__(self, target_fps: float):
        self._interval     = 1.0 / max(target_fps, 1.0)
        self._last         = time.perf_counter()
        self._fps_samples: List[float] = []

    def wait(self):
        elapsed   = time.perf_counter() - self._last
        remaining = self._interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def tick(self) -> float:
        """Record frame completion and return smoothed FPS."""
        now   = time.perf_counter()
        delta = now - self._last
        self._last = now
        if delta > 0:
            self._fps_samples.append(1.0 / delta)
            if len(self._fps_samples) > 30:
                self._fps_samples.pop(0)
        return (
            round(sum(self._fps_samples) / len(self._fps_samples), 1)
            if self._fps_samples else 0.0
        )