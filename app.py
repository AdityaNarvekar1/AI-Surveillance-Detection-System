"""
app.py — AI Surveillance System
CPU-optimized for i5, no GPU
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
import streamlit as st

from alert_system import AlertSystem
from detection import annotate_frame, run_parallel_detection, detect_objects, detect_violence
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
.summary-card{background:rgba(255,255,255,0.07);border-radius:14px;
  padding:18px 22px;margin:8px 0;border:1px solid rgba(255,255,255,0.15);}
h1{color:#00d4ff;font-weight:700;}h2,h3{color:#a0c8f0;}
.stButton>button{background:linear-gradient(90deg,#00d4ff,#7b2ff7);
  color:white;border:none;border-radius:8px;font-weight:600;padding:0.5rem 1.5rem;}
.stProgress>div>div{background:linear-gradient(90deg,#00d4ff,#7b2ff7);}
</style>""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "processing":       False,
        "show_summary":     False,
        "alert_system":     AlertSystem(cooldown_seconds=2),
        "detection_log":    [],
        "fps_display":      0.0,
        "frame_count":      0,
        "total_frames":     1,
        "last_frame_b64":   "",
        "last_progress":    0.0,
        "last_ts":          "0:00:00",
        "last_video_source": None,
        "last_ctrl":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
ensure_log_dirs()

@st.cache_resource(show_spinner="Loading AI models... (first run only)")
def _load_models():
    return load_all_models()

if "result_queue" not in st.session_state:
    st.session_state["result_queue"] = queue.Queue(maxsize=4)
if "stop_event" not in st.session_state:
    st.session_state["stop_event"] = threading.Event()


# ── Processing thread ──────────────────────────────────────────────────────────
def _processing_loop(video_source, models, alert_system,
                     yolo_conf, violence_conf, frame_skip,
                     save_snapshots, result_q, stop_event,
                     use_weapons, use_fire, use_violence):
    """
    CPU-optimized loop with per-model enable toggles.
    - YOLO runs every YOLO_EVERY frames
    - Violence runs every VIOLENCE_EVERY frames
    - Only enabled models are called
    """
    YOLO_EVERY     = max(frame_skip * 2, 4)
    VIOLENCE_EVERY = max(frame_skip, 2)
    JPEG_QUALITY   = 45
    INFER_SIZE     = (320, 240)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        result_q.put({"error": f"Cannot open: {video_source}"})
        return

    info         = get_video_info(cap)
    fps_ctrl     = FPSController(info["fps"])
    frame_buffer = []
    frame_idx    = 0

    last_detections = {
        "weapons": [], "fire_smoke": [],
        "violence_prob": 0.0, "is_violent": False
    }

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx    += 1
        video_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        video_ts      = format_timestamp(video_seconds)

        infer_frame = cv2.resize(frame, INFER_SIZE)
        frame_buffer.append(infer_frame)
        if len(frame_buffer) > 8:
            frame_buffer.pop(0)

        run_yolo     = (frame_idx % YOLO_EVERY     == 0) and (use_weapons or use_fire)
        run_violence = (frame_idx % VIOLENCE_EVERY == 0) and use_violence

        if run_yolo or run_violence:
            import concurrent.futures

            futures = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                if run_yolo and use_weapons:
                    futures["weapons"] = ex.submit(
                        detect_objects, infer_frame,
                        models["guns_knives_model"], yolo_conf)
                if run_yolo and use_fire:
                    futures["fire"] = ex.submit(
                        detect_objects, infer_frame,
                        models["fire_smoke_model"], yolo_conf)
                if run_violence:
                    futures["violence"] = ex.submit(
                        detect_violence, frame_buffer,
                        models["violence_model"],
                        models["violence_input_shape"],
                        violence_conf)

                weapons    = futures["weapons"].result()  if "weapons"  in futures else last_detections["weapons"]
                fire_smoke = futures["fire"].result()     if "fire"     in futures else last_detections["fire_smoke"]
                if "violence" in futures:
                    v_prob, is_violent = futures["violence"].result()
                else:
                    v_prob, is_violent = last_detections["violence_prob"], last_detections["is_violent"]

            # If model toggled OFF, clear its results
            if not use_weapons:  weapons    = []
            if not use_fire:     fire_smoke = []
            if not use_violence: v_prob, is_violent = 0.0, False

            last_detections = {
                "weapons":       weapons,
                "fire_smoke":    fire_smoke,
                "violence_prob": v_prob,
                "is_violent":    is_violent,
            }

        detections = dict(last_detections)

        # ── Alerts ────────────────────────────────────────────────────────
        new_alerts = []
        if run_yolo or run_violence:
            if use_violence and detections["is_violent"]:
                a = alert_system.check_and_raise("Violence", video_ts, detections["violence_prob"])
                if a:
                    new_alerts.append(a)
                    log_event("Violence", video_ts, detections["violence_prob"])
                    if save_snapshots:
                        save_snapshot(infer_frame, video_ts, "Violence")

            if use_weapons:
                for det in detections["weapons"]:
                    label = det["label"].capitalize()
                    a = alert_system.check_and_raise(label, video_ts, det["confidence"])
                    if a:
                        new_alerts.append(a)
                        log_event(label, video_ts, det["confidence"], str(det["box"]))
                        if save_snapshots:
                            save_snapshot(infer_frame, video_ts, label)

            if use_fire:
                for det in detections["fire_smoke"]:
                    label = det["label"].capitalize()
                    a = alert_system.check_and_raise(label, video_ts, det["confidence"])
                    if a:
                        new_alerts.append(a)
                        log_event(label, video_ts, det["confidence"], str(det["box"]))
                        if save_snapshots:
                            save_snapshot(infer_frame, video_ts, label)

        # ── Annotate + encode ─────────────────────────────────────────────
        display_frame = cv2.resize(frame, (640, 360))
        annotated     = annotate_frame(display_frame, detections,
                                       detections["violence_prob"],
                                       detections["is_violent"])
        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
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

        while not result_q.empty():
            try: result_q.get_nowait()
            except: pass
        try: result_q.put_nowait(payload)
        except: pass

        fps_ctrl.wait()

    cap.release()
    result_q.put({"done": True})


# ── Summary modal ──────────────────────────────────────────────────────────────
def show_summary_modal():
    """Full-width summary shown after video completes."""
    import pandas as pd
    alert_sys    = st.session_state["alert_system"]
    total_alerts = len(alert_sys.history)
    all_records  = list(alert_sys.history)

    # Count by type
    counts     = {}
    total_conf = {}
    for rec in all_records:
        t = rec.event_type
        counts[t]     = counts.get(t, 0) + 1
        total_conf[t] = total_conf.get(t, 0.0) + rec.confidence
    avg_conf = {t: total_conf[t] / counts[t] for t in counts}

    st.markdown("---")
    st.markdown("## 📊 Video Analysis Complete — Summary")

    # ── Explanation banner ────────────────────────────────────────────────────
    if total_alerts == 0:
        st.success("✅ No threats detected in this video.")
    else:
        unique_types = list(counts.keys())
        type_str = ", ".join(unique_types)
        st.info(
            f"**{total_alerts} total alerts** were raised across "
            f"**{len(unique_types)} threat type(s)**: {type_str}.  \n"
            f"The table below shows one row **per threat type** — "
            f"each row summarises all alerts of that type."
        )

    # ── Metric cards — one per threat type ───────────────────────────────────
    all_types  = ["Violence", "Gun", "Knife", "Fire", "Smoke"]
    emoji_map  = {"Violence":"🚨","Gun":"🔫","Knife":"🔪","Fire":"🔥","Smoke":"💨"}
    # Show Total + one card per detected type (up to 5)
    card_types = [t for t in all_types if t in counts] +                  [t for t in counts if t not in all_types]
    cols = st.columns(len(card_types) + 1)
    cols[0].metric("🚨 Total Alerts", total_alerts,
                   help="Sum of all alerts across all threat types")
    for i, t in enumerate(card_types):
        emoji = emoji_map.get(t, "⚠️")
        cols[i+1].metric(
            f"{emoji} {t}",
            counts[t],
            help=f"Avg confidence: {avg_conf[t]:.0%} · Max: {max(r.confidence for r in all_records if r.event_type==t):.0%}"
        )

    # ── Breakdown table — one row per threat type ─────────────────────────────
    st.markdown("### 📋 Alert Breakdown")
    st.caption("One row per threat type. Each row summarises ALL alerts of that type detected in the video.")
    if counts:
        rows = []
        for t in card_types:
            c = counts[t]
            confs = [r.confidence for r in all_records if r.event_type == t]
            rows.append({
                "Threat Type":    t,
                "Alert Count":    c,
                "Avg Confidence": f"{avg_conf[t]:.1%}",
                "Max Confidence": f"{max(confs):.1%}",
                "Min Confidence": f"{min(confs):.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Timeline chart — alerts over video time ───────────────────────────────
    if all_records:
        st.markdown("### ⏱ Alert Timeline")
        st.caption("When each alert was triggered during the video.")
        timeline_rows = []
        for rec in all_records:
            # Convert HH:MM:SS to seconds for plotting
            parts = rec.video_timestamp.replace("0:","").split(":")
            try:
                if len(parts) == 3:
                    secs = int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
                elif len(parts) == 2:
                    secs = int(parts[0])*60 + float(parts[1])
                else:
                    secs = float(parts[0])
            except Exception:
                secs = 0.0
            timeline_rows.append({
                "Time (seconds)": round(secs, 1),
                "Threat Type":    rec.event_type,
                "Confidence":     round(rec.confidence, 3),
            })
        df_timeline = pd.DataFrame(timeline_rows)
        st.scatter_chart(df_timeline, x="Time (seconds)", y="Confidence",
                         color="Threat Type", height=220)

    st.markdown("### 🎬 Actions")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🔁 Run Again (same video)", use_container_width=True, key="btn_run_again"):
            # Restart with same settings
            st.session_state["show_summary"] = False
            _start_processing(
                st.session_state["last_video_source"],
                st.session_state["last_ctrl"])
            st.rerun()

    with c2:
        if st.button("📂 Run New Video", use_container_width=True, key="btn_new_video"):
            st.session_state["show_summary"]   = False
            st.session_state["last_frame_b64"] = ""
            st.session_state["detection_log"]  = []
            st.session_state["alert_system"]   = AlertSystem(cooldown_seconds=2)
            st.rerun()

    with c3:
        log_path = get_log_path()
        if log_path.exists():
            with open(log_path, "rb") as f:
                st.download_button(
                    "⬇ Download Alerts CSV",
                    data=f,
                    file_name="detections.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_download_summary")


# ── Helper: start processing ───────────────────────────────────────────────────
def _start_processing(video_source, ctrl):
    st.session_state["result_queue"] = queue.Queue(maxsize=4)
    st.session_state["stop_event"]   = threading.Event()

    alert_sys = AlertSystem(cooldown_seconds=2)
    st.session_state["alert_system"]     = alert_sys
    st.session_state["detection_log"]    = []
    st.session_state["processing"]       = True
    st.session_state["show_summary"]     = False
    st.session_state["frame_count"]      = 0
    st.session_state["last_frame_b64"]   = ""
    st.session_state["last_progress"]    = 0.0
    st.session_state["last_ts"]          = "0:00:00"
    st.session_state["last_video_source"] = video_source
    st.session_state["last_ctrl"]        = ctrl

    t = threading.Thread(
        target=_processing_loop,
        args=(video_source,
              _load_models(),
              alert_sys,
              ctrl["yolo_conf"],
              ctrl["violence_conf"],
              ctrl["frame_skip"],
              ctrl["save_snaps"],
              st.session_state["result_queue"],
              st.session_state["stop_event"],
              ctrl["use_weapons"],
              ctrl["use_fire"],
              ctrl["use_violence"]),
        daemon=True)
    t.start()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(models):
    with st.sidebar:
        st.markdown("## 🔍 Control Panel")
        st.divider()

        # ── Video source ──────────────────────────────────────────────────
        st.markdown("### 📁 Video Source")
        source_type = st.radio("Input type",
                               ["Upload Video File", "Stream URL / Webcam"],
                               key="source_type", label_visibility="collapsed")
        uploaded_file = None
        stream_url    = "0"
        if source_type == "Upload Video File":
            uploaded_file = st.file_uploader("Upload a video",
                type=["mp4","avi","mov","mkv","webm"], key="video_upload")
        else:
            stream_url = st.text_input("Stream URL / Webcam index",
                                       value="0", key="stream_url")

        st.divider()

        # ── Detection thresholds ──────────────────────────────────────────
        st.markdown("### ⚙️ Detection Thresholds")
        st.caption("**YOLO Confidence** — min certainty to draw a bounding box")
        yolo_conf = st.slider("YOLO Confidence", 0.1, 1.0, 0.45, 0.05,
                              key="yolo_conf", label_visibility="collapsed")
        st.caption("**Violence Threshold** — min probability to trigger violence alert")
        violence_conf = st.slider("Violence Threshold", 0.1, 1.0, 0.65, 0.05,
                                  key="violence_conf", label_visibility="collapsed")

        st.divider()

        # ── Model toggles ─────────────────────────────────────────────────
        st.markdown("### 🤖 Active Models")
        st.caption("Toggle which models run on this video")

        v_ok  = models.get("violence_model")    is not None
        gk_ok = models.get("guns_knives_model") is not None
        fs_ok = models.get("fire_smoke_model")  is not None

        col_a, col_b = st.columns([3, 1])
        with col_a: st.markdown("🔫 Guns & Knives")
        with col_b: use_weapons = st.toggle("", value=gk_ok, key="tog_weapons",
                                             disabled=not gk_ok)

        col_a, col_b = st.columns([3, 1])
        with col_a: st.markdown("🔥 Fire & Smoke")
        with col_b: use_fire = st.toggle("", value=fs_ok, key="tog_fire",
                                          disabled=not fs_ok)

        col_a, col_b = st.columns([3, 1])
        with col_a: st.markdown("🚨 Violence Detection")
        with col_b: use_violence = st.toggle("", value=v_ok, key="tog_violence",
                                              disabled=not v_ok)

        if not use_weapons and not use_fire and not use_violence:
            st.warning("⚠️ Enable at least one model.")

        st.divider()

        # ── Performance ───────────────────────────────────────────────────
        st.markdown("### 🚀 Performance")
        st.caption("**Frame Skip** — process 1 in every N frames")
        frame_skip = st.selectbox("Process every Nth frame", [2, 3, 4, 6],
                                  index=1, key="frame_skip",
                                  label_visibility="collapsed",
                                  help="3 recommended on i5 CPU")
        save_snaps = st.checkbox("Save snapshots on detection",
                                 value=True, key="save_snaps")
        st.info("💡 Frame skip 3 recommended for your i5 CPU.")

        st.divider()

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

    return dict(
        uploaded_file=uploaded_file, source_type=source_type,
        stream_url=stream_url, yolo_conf=yolo_conf,
        violence_conf=violence_conf, frame_skip=frame_skip,
        save_snaps=save_snaps, start=start_btn, stop=stop_btn,
        use_weapons=use_weapons, use_fire=use_fire, use_violence=use_violence)


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
    m1.metric("⚡ FPS",   f"{st.session_state.get('fps_display', 0):.1f}")
    m2.metric("🎞 Frame",
              f"{st.session_state.get('frame_count',0)} / "
              f"{st.session_state.get('total_frames',1)}")
    m3.metric("🚨 Alerts", len(st.session_state["alert_system"].history))
    m4.metric("Status", "🟢 Running" if st.session_state["processing"] else "⚫ Idle")
    st.divider()

    # ── Summary modal (shown after completion) ────────────────────────────────
    if st.session_state.get("show_summary"):
        show_summary_modal()
        return   # don't show live feed while summary is up

    # ── Live feed layout ──────────────────────────────────────────────────────
    vid_col, info_col = st.columns([3, 2])

    with vid_col:
        st.markdown("### 📺 Live Feed")
        st.progress(st.session_state.get("last_progress", 0.0))
        st.caption(f"⏱ Video time: **{st.session_state.get('last_ts','0:00:00')}**")

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
        # Always reference session_state directly — never cache in local var across reruns
        recent = st.session_state["alert_system"].recent_alerts(8)
        if recent:
            html = ""
            for a in recent:
                c = "#ff4444" if st.session_state["alert_system"].severity(a["event_type"]) == "error" else "#ffaa00"
                html += (f"<div class='alert-box' style='border-color:{c};'>"
                         f"{a['message']}</div>")
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No alerts yet.")

        st.markdown("### 📋 Detection Log")
        import pandas as pd
        # Use exact same method as alert panel — recent_alerts() from alert_system
        alert_sys_ref = st.session_state["alert_system"]
        log_data = alert_sys_ref.recent_alerts(50)  # returns list of dicts
        if log_data:
            df = pd.DataFrame(log_data)
            # Show all available columns safely
            show_cols = [c for c in ["event_type","video_timestamp","confidence","message"]
                         if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.markdown("_No detections yet._")

    # ── Start ─────────────────────────────────────────────────────────────────
    if ctrl["start"] and not st.session_state["processing"]:
        if not (ctrl["use_weapons"] or ctrl["use_fire"] or ctrl["use_violence"]):
            st.error("Please enable at least one model before starting.")
            st.stop()

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

        _start_processing(video_source, ctrl)
        time.sleep(0.5)
        st.rerun()

    # ── Stop ──────────────────────────────────────────────────────────────────
    if ctrl["stop"] and st.session_state["processing"]:
        st.session_state["stop_event"].set()
        st.session_state["processing"] = False
        st.rerun()

    # ── Poll + rerun ───────────────────────────────────────────────────────────
    if st.session_state["processing"]:
        result_q = st.session_state["result_queue"]

        payload = None
        while True:
            try: payload = result_q.get_nowait()
            except queue.Empty: break

        if payload is None:
            time.sleep(0.4)
            st.rerun()

        elif "error" in payload:
            st.error(payload["error"])
            st.session_state["processing"] = False
            st.rerun()

        elif "done" in payload:
            # Video finished — show summary
            st.session_state["processing"]  = False
            st.session_state["show_summary"] = True
            st.rerun()

        else:
            st.session_state["last_frame_b64"] = payload["frame_b64"]
            st.session_state["last_progress"]  = min(
                payload["frame_idx"] / max(payload["total_frames"], 1), 1.0)
            st.session_state["last_ts"]        = payload["video_ts"]
            st.session_state["fps_display"]    = payload["fps"]
            st.session_state["frame_count"]    = payload["frame_idx"]
            st.session_state["total_frames"]   = payload["total_frames"]

            # Sync detection_log directly from alert_system.history
            # This ensures nothing is lost even if UI polls slowly
            st.session_state["detection_log"] = [
                r.to_dict() for r in reversed(list(
                    st.session_state["alert_system"].history))
            ]

            time.sleep(0.35)
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#445566;font-size:0.8rem;'>"
        "🛡️ AI Surveillance System · YOLOv8 + Deep Learning</p>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()