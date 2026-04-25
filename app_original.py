"""
app.py — AI Surveillance System
Run: streamlit run app.py
"""

import base64
import logging
import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from alert_system import AlertSystem
from detection import annotate_frame, run_parallel_detection
from model_loader import load_all_models
from utils import (
    FPSController, ensure_log_dirs, format_timestamp,
    get_log_path, get_video_info, log_event, save_snapshot,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Surveillance System", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#e0e0e0;}
section[data-testid="stSidebar"]{background:rgba(255,255,255,0.05);
  backdrop-filter:blur(12px);border-right:1px solid rgba(255,255,255,0.1);}
[data-testid="metric-container"]{background:rgba(255,255,255,0.07);
  border-radius:12px;padding:10px;border:1px solid rgba(255,255,255,0.12);}
.alert-box{background:rgba(255,50,50,0.15);border-left:4px solid #ff4444;
  border-radius:8px;padding:10px 14px;margin:6px 0;font-size:0.9rem;}
h1{color:#00d4ff;font-weight:700;}h2,h3{color:#a0c8f0;}
.stButton>button{background:linear-gradient(90deg,#00d4ff,#7b2ff7);
  color:white;border:none;border-radius:8px;font-weight:600;padding:0.5rem 1.5rem;}
.stProgress>div>div{background:linear-gradient(90deg,#00d4ff,#7b2ff7);}
</style>""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "processing":       False,
        "alert_system":     AlertSystem(cooldown_seconds=2),
        "detection_log":    [],
        "fps_display":      0.0,
        "frame_count":      0,
        "total_frames":     1,
        "last_frame_b64":   "",
        "last_progress":    0.0,
        "last_ts":          "0:00:00",
        "last_alerts_html": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
ensure_log_dirs()

@st.cache_resource(show_spinner="Loading AI models... (first run only)")
def _load_models():
    return load_all_models()

# ── Shared thread communication (module-level, survives reruns) ────────────────
if "result_queue" not in st.session_state:
    st.session_state["result_queue"] = queue.Queue(maxsize=8)
if "stop_event" not in st.session_state:
    st.session_state["stop_event"] = threading.Event()

# ── Processing thread ──────────────────────────────────────────────────────────
def _processing_loop(video_source, models, alert_system,
                     yolo_conf, violence_conf, frame_skip,
                     save_snapshots, result_q, stop_event):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        result_q.put({"error": f"Cannot open: {video_source}"})
        return

    info = get_video_info(cap)
    fps_ctrl = FPSController(info["fps"])
    frame_buffer = []
    frame_idx = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        video_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        video_ts = format_timestamp(video_seconds)

        if frame_idx % max(frame_skip, 1) != 0:
            fps_ctrl.wait()
            fps_ctrl.tick()
            continue

        display_frame = cv2.resize(frame, (640, 360))
        frame_buffer.append(display_frame)
        if len(frame_buffer) > 32:
            frame_buffer.pop(0)

        detections = run_parallel_detection(
            display_frame, frame_buffer,
            models["guns_knives_model"], models["fire_smoke_model"],
            models["violence_model"], models["violence_input_shape"],
            yolo_conf=yolo_conf, violence_conf=violence_conf,
        )

        new_alerts = []

        if detections["is_violent"]:
            a = alert_system.check_and_raise("Violence", video_ts, detections["violence_prob"])
            if a:
                new_alerts.append(a)
                log_event("Violence", video_ts, detections["violence_prob"])
                if save_snapshots:
                    save_snapshot(display_frame, video_ts, "Violence")

        for det in detections["weapons"]:
            label = det["label"].capitalize()
            a = alert_system.check_and_raise(label, video_ts, det["confidence"])
            if a:
                new_alerts.append(a)
                log_event(label, video_ts, det["confidence"], str(det["box"]))
                if save_snapshots:
                    save_snapshot(display_frame, video_ts, label)

        for det in detections["fire_smoke"]:
            label = det["label"].capitalize()
            a = alert_system.check_and_raise(label, video_ts, det["confidence"])
            if a:
                new_alerts.append(a)
                log_event(label, video_ts, det["confidence"], str(det["box"]))
                if save_snapshots:
                    save_snapshot(display_frame, video_ts, label)

        annotated = annotate_frame(display_frame, detections,
                                   detections["violence_prob"], detections["is_violent"])
        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(jpg.tobytes()).decode()

        fps = fps_ctrl.tick()

        payload = {
            "frame_b64":    b64,
            "fps":          fps,
            "frame_idx":    frame_idx,
            "total_frames": info["total_frames"],
            "video_ts":     video_ts,
            "new_alerts":   [a.to_dict() for a in new_alerts],
        }

        # Always keep only the latest frame — drain old ones first
        while not result_q.empty():
            try: result_q.get_nowait()
            except: pass
        try: result_q.put_nowait(payload)
        except: pass

        fps_ctrl.wait()

    cap.release()
    result_q.put({"done": True})


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(models):
    with st.sidebar:
        st.markdown("## 🔍 Control Panel")
        st.divider()
        st.markdown("### 📁 Video Source")
        source_type = st.radio("Input type",
                               ["Upload Video File", "Stream URL / Webcam"],
                               key="source_type", label_visibility="collapsed")
        uploaded_file = None
        stream_url = "0"
        if source_type == "Upload Video File":
            uploaded_file = st.file_uploader("Upload a video",
                type=["mp4","avi","mov","mkv","webm"], key="video_upload")
        else:
            stream_url = st.text_input("Stream URL / Webcam index",
                                       value="0", key="stream_url")

        st.divider()
        st.markdown("### ⚙️ Detection Thresholds")
        yolo_conf     = st.slider("YOLO Confidence",    0.1, 1.0, 0.4, 0.05, key="yolo_conf")
        violence_conf = st.slider("Violence Threshold", 0.1, 1.0, 0.6, 0.05, key="violence_conf")

        st.divider()
        st.markdown("### 🚀 Performance")
        frame_skip = st.selectbox("Process every Nth frame", [1,2,3,4],
                                  index=1, key="frame_skip",
                                  help="2-3 recommended on CPU")
        save_snaps = st.checkbox("Save snapshots on detection",
                                 value=True, key="save_snaps")

        st.divider()
        st.markdown("### 🤖 Model Status")
        v_ok  = models.get("violence_model")    is not None
        gk_ok = models.get("guns_knives_model") is not None
        fs_ok = models.get("fire_smoke_model")  is not None
        st.markdown(
            f"{'✅' if gk_ok else '❌'} Guns & Knives (YOLO)  \n"
            f"{'✅' if fs_ok else '❌'} Fire & Smoke (YOLO)   \n"
            f"{'✅' if v_ok  else '⚠️'} Violence Model (Keras)")

        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶ Start", key="start_btn", use_container_width=True)
        with col2:
            stop_btn  = st.button("⏹ Stop",  key="stop_btn",  use_container_width=True)

        st.divider()
        log_path = get_log_path()
        if log_path.exists():
            with open(log_path, "rb") as f:
                st.download_button("⬇ Download CSV Log", data=f,
                    file_name="detections.csv", mime="text/csv",
                    use_container_width=True)

    return dict(uploaded_file=uploaded_file, source_type=source_type,
                stream_url=stream_url, yolo_conf=yolo_conf,
                violence_conf=violence_conf, frame_skip=frame_skip,
                save_snaps=save_snaps, start=start_btn, stop=stop_btn)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px 0;'>
      <h1 style='margin:0;'>🛡️ AI Surveillance System</h1>
      <p style='color:#8899aa;font-size:1.05rem;margin-top:4px;'>
        Real-Time Violence &amp; Anomaly Detection · YOLOv8 &amp; Deep Learning
      </p>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading AI models..."):
        models = _load_models()

    ctrl = render_sidebar(models)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⚡ FPS",    f"{st.session_state.get('fps_display', 0):.1f}")
    m2.metric("🎞 Frame",
              f"{st.session_state.get('frame_count',0)} / "
              f"{st.session_state.get('total_frames',1)}")
    m3.metric("🚨 Alerts", len(st.session_state["alert_system"].history))
    m4.metric("Status", "🟢 Running" if st.session_state["processing"] else "⚫ Idle")
    st.divider()

    # ── Layout ────────────────────────────────────────────────────────────────
    vid_col, info_col = st.columns([3, 2])

    with vid_col:
        st.markdown("### 📺 Live Feed")
        progress_bar = st.progress(st.session_state.get("last_progress", 0.0))
        st.caption(f"⏱ Video time: **{st.session_state.get('last_ts','0:00:00')}**")

        # Show last known frame, or placeholder
        if st.session_state.get("last_frame_b64"):
            st.markdown(
                f'<img src="data:image/jpeg;base64,{st.session_state["last_frame_b64"]}" '
                f'style="width:100%;border-radius:8px;">',
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='height:360px;display:flex;align-items:center;"
                "justify-content:center;background:rgba(0,0,0,0.4);"
                "border-radius:12px;color:#555;font-size:1.1rem;'>"
                "Upload a video and press ▶ Start</div>",
                unsafe_allow_html=True)

    with info_col:
        st.markdown("### 🚨 Alert Panel")
        alert_sys = st.session_state["alert_system"]
        recent = alert_sys.recent_alerts(8)
        if recent:
            html = ""
            for a in recent:
                c = "#ff4444" if alert_sys.severity(a["event_type"]) == "error" else "#ffaa00"
                html += (f"<div class='alert-box' style='border-color:{c};'>"
                         f"{a['message']}</div>")
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No alerts yet.")

        st.markdown("### 📋 Detection Log")
        log_data = st.session_state["detection_log"][:50]
        if log_data:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(log_data)[
                    ["event_type","video_timestamp","confidence","message"]],
                use_container_width=True, hide_index=True)
        else:
            st.markdown("_No detections yet._")

    # ── Start ─────────────────────────────────────────────────────────────────
    if ctrl["start"] and not st.session_state["processing"]:
        if ctrl["source_type"] == "Upload Video File":
            if ctrl["uploaded_file"] is None:
                st.error("Please upload a video file first.")
                st.stop()
            suffix = Path(ctrl["uploaded_file"].name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(ctrl["uploaded_file"].read())
            tmp.flush(); tmp.close()
            video_source = tmp.name
        else:
            raw = ctrl["stream_url"].strip()
            video_source = int(raw) if raw.isdigit() else raw

        # Fresh queue and stop event each run
        st.session_state["result_queue"] = queue.Queue(maxsize=8)
        st.session_state["stop_event"]   = threading.Event()

        alert_sys = AlertSystem(cooldown_seconds=2)
        st.session_state["alert_system"]   = alert_sys
        st.session_state["detection_log"]  = []
        st.session_state["processing"]     = True
        st.session_state["frame_count"]    = 0
        st.session_state["last_frame_b64"] = ""
        st.session_state["last_progress"]  = 0.0
        st.session_state["last_ts"]        = "0:00:00"

        t = threading.Thread(
            target=_processing_loop,
            args=(video_source, models, alert_sys,
                  ctrl["yolo_conf"], ctrl["violence_conf"],
                  ctrl["frame_skip"], ctrl["save_snaps"],
                  st.session_state["result_queue"],
                  st.session_state["stop_event"]),
            daemon=True)
        t.start()
        time.sleep(0.5)   # let thread start up before first rerun
        st.rerun()

    # ── Stop ──────────────────────────────────────────────────────────────────
    if ctrl["stop"] and st.session_state["processing"]:
        st.session_state["stop_event"].set()
        st.session_state["processing"] = False
        st.rerun()

    # ── Poll queue and rerun ───────────────────────────────────────────────────
    if st.session_state["processing"]:
        result_q = st.session_state["result_queue"]

        # Read the latest available frame (drain stale ones)
        payload = None
        while True:
            try:
                payload = result_q.get_nowait()
            except queue.Empty:
                break

        if payload is None:
            # No frame yet — wait briefly and rerun to poll again
            time.sleep(0.3)
            st.rerun()

        elif "error" in payload:
            st.error(payload["error"])
            st.session_state["processing"] = False
            st.rerun()

        elif "done" in payload:
            st.session_state["processing"] = False
            st.success("✅ Processing complete!")
            st.rerun()

        else:
            # Store latest frame and state in session_state, then rerun to redraw
            st.session_state["last_frame_b64"] = payload["frame_b64"]
            st.session_state["last_progress"]  = min(
                payload["frame_idx"] / max(payload["total_frames"], 1), 1.0)
            st.session_state["last_ts"]        = payload["video_ts"]
            st.session_state["fps_display"]    = payload["fps"]
            st.session_state["frame_count"]    = payload["frame_idx"]
            st.session_state["total_frames"]   = payload["total_frames"]

            for a in payload["new_alerts"]:
                st.session_state["detection_log"].insert(0, a)

            time.sleep(0.25)  # ~4 UI updates/sec — stable, no WebSocket overload
            st.rerun()


if __name__ == "__main__":
    main()