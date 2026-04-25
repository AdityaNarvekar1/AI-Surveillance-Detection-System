"""
detection.py
------------
Per-frame detection logic for all three models.

Public API
----------
detect_objects(frame, yolo_model, conf_threshold)               → list[dict]
detect_violence(frame_buffer, violence_model, input_shape, conf) → (prob, is_violent)
run_parallel_detection(...)                                      → dict
annotate_frame(frame, detections, violence_prob, is_violent)    → annotated_frame
"""

import concurrent.futures
import logging

import cv2
import numpy as np

from utils import preprocess_frame, build_sequence_input

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
_COLOURS = {
    "gun":     (0,   0, 255),   # red
    "knife":   (0, 165, 255),   # orange
    "fire":    (0,  50, 255),   # deep orange-red
    "smoke":   (180, 180, 180), # grey
    "default": (0,  255,   0),  # green
}


def _colour_for(label: str) -> tuple:
    label_l = label.lower()
    for key, colour in _COLOURS.items():
        if key in label_l:
            return colour
    return _COLOURS["default"]


# ── YOLO detection ─────────────────────────────────────────────────────────────

def detect_objects(frame: np.ndarray, yolo_model, conf_threshold: float = 0.4) -> list:
    """
    Run a YOLOv8 model on a single frame.

    Returns
    -------
    list of dicts: {label, confidence, box: (x1, y1, x2, y2)}
    """
    if yolo_model is None:
        return []
    try:
        results = yolo_model(frame, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = r.names.get(cls_id, str(cls_id))
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label":      label,
                    "confidence": conf,
                    "box":        (x1, y1, x2, y2),
                })
        return detections
    except Exception as e:
        logger.error("YOLO inference error: %s", e)
        return []


# ── Violence detection ─────────────────────────────────────────────────────────

def _infer_violence_mode(input_shape) -> str:
    """
    Determine whether model expects:
      'single'   → (batch, H, W, 3)
      'sequence' → (batch, T, H, W, 3)
    """
    if input_shape is None:
        return "single"
    return "sequence" if len(input_shape) == 5 else "single"


def detect_violence(
    frame_buffer: list,
    violence_model,
    input_shape,
    conf_threshold: float = 0.6,
) -> tuple:
    """
    Run the Keras violence model on the rolling frame buffer.

    Returns
    -------
    (probability: float, is_violent: bool)
    """
    if violence_model is None or not frame_buffer:
        return 0.0, False

    try:
        mode = _infer_violence_mode(input_shape)

        if mode == "sequence":
            # Expected shape: (None, T, H, W, 3)
            T = input_shape[1] if (input_shape[1] is not None) else 16
            H = input_shape[2] if (input_shape[2] is not None) else 224
            W = input_shape[3] if (input_shape[3] is not None) else 224
            inp = build_sequence_input(frame_buffer, T, size=(W, H))
        else:
            # Expected shape: (None, H, W, 3)
            H = input_shape[1] if (input_shape and input_shape[1] is not None) else 224
            W = input_shape[2] if (input_shape and input_shape[2] is not None) else 224
            proc = preprocess_frame(frame_buffer[-1], size=(W, H))
            inp  = np.expand_dims(proc, axis=0)   # (1, H, W, 3)

        preds = violence_model.predict(inp, verbose=0)
        preds_flat = preds.flatten()

        if len(preds_flat) == 1:
            # Binary sigmoid output
            prob = float(preds_flat[0])
        elif len(preds_flat) == 2:
            # Softmax: [non_violent, violent]
            prob = float(preds_flat[1])
        else:
            # Multi-class: take max
            prob = float(preds_flat.max())

        return prob, prob >= conf_threshold

    except Exception as e:
        logger.error("Violence inference error: %s", e)
        return 0.0, False


# ── Parallel inference ─────────────────────────────────────────────────────────

def run_parallel_detection(
    frame: np.ndarray,
    frame_buffer: list,
    guns_knives_model,
    fire_smoke_model,
    violence_model,
    violence_input_shape,
    yolo_conf: float = 0.4,
    violence_conf: float = 0.6,
) -> dict:
    """
    Run all three models concurrently using a thread pool.

    Returns
    -------
    {
        "weapons":       list[dict],
        "fire_smoke":    list[dict],
        "violence_prob": float,
        "is_violent":    bool,
    }
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_weapons  = executor.submit(detect_objects, frame, guns_knives_model, yolo_conf)
        fut_fire     = executor.submit(detect_objects, frame, fire_smoke_model,  yolo_conf)
        fut_violence = executor.submit(
            detect_violence, frame_buffer, violence_model, violence_input_shape, violence_conf
        )

        weapons    = fut_weapons.result()
        fire_smoke = fut_fire.result()
        v_prob, is_violent = fut_violence.result()

    return {
        "weapons":       weapons,
        "fire_smoke":    fire_smoke,
        "violence_prob": v_prob,
        "is_violent":    is_violent,
    }


# ── Frame annotation ───────────────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    detections: dict,
    violence_prob: float = 0.0,
    is_violent: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes and overlays on the frame.
    Returns an annotated copy — does NOT modify original.
    """
    out = frame.copy()
    all_objects = detections.get("weapons", []) + detections.get("fire_smoke", [])

    for det in all_objects:
        x1, y1, x2, y2 = det["box"]
        label  = det["label"]
        conf   = det["confidence"]
        colour = _colour_for(label)

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        # Label with background
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Violence probability bar (top-left)
    bar_colour = (0, 0, 220) if is_violent else (0, 200, 0)
    label_text = f"Violence: {violence_prob:.2f}"
    cv2.rectangle(out, (10, 10), (260, 40), (0, 0, 0), -1)
    cv2.putText(out, label_text, (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, bar_colour, 2)

    # Red border when any threat detected
    if is_violent or all_objects:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    return out