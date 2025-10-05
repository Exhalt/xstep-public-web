# app.py (knee-direction flip before processing)
import os, io, json, tempfile, time, subprocess, base64, re, sys, asyncio, textwrap, urllib.request, hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import mediapipe as mp

import xstep_core as xstep  # your core pipeline

st.set_page_config(page_title="Xstep – Running Posture Analysis", layout="wide")

# ---- Windows asyncio (helps Streamlit on Win) ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ======================= constants & labels =======================
CATS_NUMERIC = ["shank","foot_strike","foot_pitch","msa","symmetry"]
CAT_LABEL = {
    "shank":"Shank/Ankle",
    "foot_strike":"Foot Strike",
    "foot_pitch":"Foot Pitch",
    "msa":"Max Shank Angle",
    "symmetry":"L/R Symmetry",
}
VIDEO_W = 480
POSE = mp.solutions.pose

# ======================= summaries & scores helpers =======================
def load_all_summaries(out_dir: Path):
    files = sorted(out_dir.glob("summary_lap_*.json"))
    summaries = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                summaries.append(json.load(fh))
        except Exception:
            pass
    return summaries

def _normalize_text(s: str) -> str:
    t = str(s)
    t = t.replace("≥", ">=")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def no_running_via_recommendation(summaries) -> bool:
    pats = [
        r"no running/jogging segments.*detected in this video",
        r"no running/jogging segments \(>=\s*\d+\s*s\) detected in this video",
    ]
    for s in summaries:
        recs = s.get("recommendations") or s.get("recommendation") or []
        if isinstance(recs, (str, int, float)):
            recs = [str(recs)]
        for r in recs:
            txt = _normalize_text(r)
            for pat in pats:
                if re.search(pat, txt):
                    return True
    return False

def aggregate_numeric_scores(summaries):
    agg = {k:0.0 for k in CATS_NUMERIC}
    cnt = {k:0 for k in CATS_NUMERIC}
    recs = []
    for s in summaries:
        seg_raw = s.get("segment_raw", {})
        for k in CATS_NUMERIC:
            try: v = float(seg_raw.get(k, 0.0))
            except Exception: v = 0.0
            agg[k] += v; cnt[k] += 1
        r = s.get("recommendations", [])
        recs.extend([str(x) for x in r if isinstance(x, (str,int,float,str))])
    avg = {k:(agg[k]/max(1,cnt[k])) for k in CATS_NUMERIC}
    return avg, recs

def derive_posture_label_stats(summaries):
    totals = {"Straight":0, "Curved forward":0, "Arched (backward)":0}
    for s in summaries:
        dist = s.get("posture_distribution") or {}
        for k in totals.keys():
            try:
                totals[k] += int(dist.get(k, 0))
            except Exception:
                pass
    total_frames = sum(totals.values())
    if total_frames <= 0:
        return "-", {k:0.0 for k in totals.keys()}
    majority = max(totals.items(), key=lambda kv: kv[1])[0]
    pct = {k: round(100.0*totals[k]/total_frames, 1) for k in totals.keys()}
    return majority, pct

def norm_to_10(raw_dict):
    out = {}
    for k,v in raw_dict.items():
        try: v = float(v)
        except Exception: v = 0.0
        v = max(-5.0, min(5.0, v))
        out[k] = round((v+5.0)/10.0*10.0, 1)
    return out

def zeros_scores():
    return {k:0.0 for k in CATS_NUMERIC}

# ======================= plotting & video helpers =======================
def make_radar(scores_0_10: dict):
    axes = [CAT_LABEL[k] for k in CATS_NUMERIC]
    vals = [scores_0_10.get(k,0.0) for k in CATS_NUMERIC]
    axes.append(axes[0]); vals.append(vals[0])
    fig = go.Figure(data=go.Scatterpolar(r=vals, theta=axes, fill='toself'))
    fig.update_layout(margin=dict(l=30,r=30,t=30,b=30),
                      polar=dict(radialaxis=dict(range=[0,10])))
    return fig

def render_video_compact(video_bytes: bytes, width_px: int = VIDEO_W):
    b64 = base64.b64encode(video_bytes).decode("utf-8")
    html = f"""
    <div style="max-width:{width_px}px;width:100%;margin:0;">
      <video src="data:video/mp4;base64,{b64}" controls
        style="width:100%;height:auto;display:block;border-radius:8px;">
      </video>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

def reencode_h264(input_mp4: Path, output_mp4: Path | None = None) -> Path:
    # Try to normalize for browser playback
    try:
        import imageio_ffmpeg
        if output_mp4 is None:
            output_mp4 = input_mp4.with_name(input_mp4.stem + "_h264.mp4")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", str(input_mp4),
            "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(output_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_mp4
    except Exception:
        return input_mp4

def extract_video_thumbnail(video_path: str, jpg_out: Path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read(); cap.release()
    if ok and frame is not None:
        cv2.imwrite(str(jpg_out), frame)
        return True
    return False

# ======================= features.csv helpers =======================
def read_features_csv(out_dir: Path) -> pd.DataFrame | None:
    path = out_dir / "features.csv"
    if not path.exists():
        return None
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception:
            continue
    return None

def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def running_nanmax(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.empty_like(arr)
    cur = np.nan
    for i, v in enumerate(arr):
        if not np.isnan(v):
            cur = v if np.isnan(cur) else max(cur, v)
        out[i] = cur
    return out

# ======================= Orientation (upright) helpers =======================
def _safe_lm(lm, thr=0.4):
    if lm is None: return None
    vis = float(getattr(lm, "visibility", 0.0))
    if vis < thr: return None
    return (lm.x, lm.y)

def detect_best_rotation(video_path: str, sample_frames=36):
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(min(total or fps*4, fps*4))
    idxs = list(range(0, max_frames, max(1, max_frames // sample_frames)))[:sample_frames]

    scores = {rot: 0 for rot in rotations}
    seen   = {rot: 0 for rot in rotations}

    with POSE.Pose(static_image_mode=False, model_complexity=1,
                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for rot in rotations:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in idxs:
                if total > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok or frame is None: continue
                if rot is not None: frame = cv2.rotate(frame, rot)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if not res.pose_landmarks: continue
                lms = res.pose_landmarks.landmark
                # head above feet and hips under shoulders
                nose   = _safe_lm(lms[POSE.PoseLandmark.NOSE.value])
                lankle = _safe_lm(lms[POSE.PoseLandmark.LEFT_ANKLE.value])
                rankle = _safe_lm(lms[POSE.PoseLandmark.RIGHT_ANKLE.value])
                lhip   = _safe_lm(lms[POSE.PoseLandmark.LEFT_HIP.value])
                rhip   = _safe_lm(lms[POSE.PoseLandmark.RIGHT_HIP.value])
                lsh    = _safe_lm(lms[POSE.PoseLandmark.LEFT_SHOULDER.value])
                rsh    = _safe_lm(lms[POSE.PoseLandmark.RIGHT_SHOULDER.value])
                heads = [p for p in (nose,) if p]; feet = [p for p in (lankle, rankle) if p]
                if not heads or not feet or not lhip or not rhip or not lsh or not rsh:
                    continue
                y_head = min(p[1] for p in heads); y_foot = max(p[1] for p in feet)
                yS = (lsh[1] + rsh[1]) / 2.0; yH = (lhip[1] + rhip[1]) / 2.0
                if (y_head + 8.0/max(h,1)) < y_foot and yH > yS:
                    seen[rot] += 1; scores[rot] += 1

    cap.release()
    best_rot = max(scores.items(), key=lambda kv: kv[1])[0]
    conf = (scores[best_rot] / max(1, seen[best_rot])) if seen[best_rot] else 0.0
    return best_rot, conf

def rotate_transcode_cv2(input_mp4: Path, output_mp4: Path, rot_code) -> Path:
    cap = cv2.VideoCapture(str(input_mp4))
    if not cap.isOpened(): return input_mp4
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, fr = cap.read()
    if not ok or fr is None: cap.release(); return input_mp4
    rot = cv2.rotate(fr, rot_code)
    H, W = rot.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_mp4), fourcc, fps, (W, H))
    if not out.isOpened(): cap.release(); return input_mp4
    out.write(rot)
    while True:
        ok, fr = cap.read()
        if not ok: break
        out.write(cv2.rotate(fr, rot_code))
    cap.release(); out.release()
    return output_mp4

def auto_upright_video(inp_path: Path) -> tuple[Path, str]:
    msg = []
    r1, c1 = detect_best_rotation(str(inp_path))
    if r1 in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180) and c1 >= 0.45:
        msg.append(f"Auto-rotating (conf {c1*100:.0f}%).")
        tmp1 = inp_path.with_name("input_rot1.mp4")
        out1 = rotate_transcode_cv2(inp_path, tmp1, r1)
        return out1, " ".join(msg)
    elif c1 < 0.4:
        msg.append("Orientation check low confidence; left as-is.")
    else:
        msg.append("Looks upright; no rotation applied.")
    return inp_path, " ".join(msg)

# ======================= NEW: Knee-based direction detection & flip =======================
def detect_run_direction_knee(video_path: str, sample_seconds: float = 2.5) -> tuple[str, float]:
    """
    Estimate left/right screen motion from knees:
      - sample ~sample_seconds
      - compute net Δx of LEFT and RIGHT knee (normalized 0..1)
      - average Δx; sign >0 => moving right, <0 => moving left
    Returns ('left'|'right'|'unknown', confidence[0..1])
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "unknown", 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*sample_seconds)
    max_frames = int(min(total, fps*sample_seconds))
    # sample ~3 frames/sec
    step = max(1, int(fps // 3))
    idxs = list(range(0, max_frames, step))

    xs_L, xs_R = [], []
    with POSE.Pose(static_image_mode=False, model_complexity=1,
                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for i in idxs:
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks: continue
            lms = res.pose_landmarks.landmark
            lk = _safe_lm(lms[POSE.PoseLandmark.LEFT_KNEE.value], thr=0.25)
            rk = _safe_lm(lms[POSE.PoseLandmark.RIGHT_KNEE.value], thr=0.25)
            if lk: xs_L.append(lk[0])  # normalized 0..1
            if rk: xs_R.append(rk[0])

    cap.release()

    # if not enough knee samples, fallback to hip-center drift
    if len(xs_L) < 4 or len(xs_R) < 4:
        cap = cv2.VideoCapture(video_path)
        xs = []
        with POSE.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for i in idxs:
                if total > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok or frame is None: continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if not res.pose_landmarks: continue
                lms = res.pose_landmarks.landmark
                lhip = _safe_lm(lms[POSE.PoseLandmark.LEFT_HIP.value], 0.25)
                rhip = _safe_lm(lms[POSE.PoseLandmark.RIGHT_HIP.value], 0.25)
                if lhip and rhip:
                    xs.append((lhip[0] + rhip[0]) / 2.0)
        cap.release()
        if len(xs) < 4:
            return "unknown", 0.0
        q = max(1, len(xs)//4)
        dx = float(np.nanmedian(xs[-q:]) - np.nanmedian(xs[:q]))
        span = float(np.nanstd(xs) + 1e-6)
        conf = float(min(1.0, max(0.0, abs(dx) / (span*2.5))))
        if dx > 0:  return "right", conf
        if dx < 0:  return "left", conf
        return "unknown", conf

    # Use *average* knee drift for robustness
    def net_delta(arr):
        q = max(1, len(arr)//4)
        return float(np.nanmedian(arr[-q:]) - np.nanmedian(arr[:q])), float(np.nanstd(arr) + 1e-6)

    dxL, sL = net_delta(xs_L)
    dxR, sR = net_delta(xs_R)
    dx_avg = (dxL + dxR) / 2.0
    span = (sL + sR) / 2.0
    conf = float(min(1.0, max(0.0, abs(dx_avg) / (span*2.5))))

    if dx_avg > 0:
        return "right", conf
    elif dx_avg < 0:
        return "left", conf
    return "unknown", conf

def flip_transcode_cv2(input_mp4: Path, output_mp4: Path) -> Path:
    cap = cv2.VideoCapture(str(input_mp4))
    if not cap.isOpened():
        return input_mp4
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, fr = cap.read()
    if not ok or fr is None:
        cap.release(); return input_mp4
    fl = cv2.flip(fr, 1)  # horizontal
    H, W = fl.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_mp4), fourcc, fps, (W, H))
    if not out.isOpened():
        cap.release(); return input_mp4
    out.write(fl)
    while True:
        ok, fr = cap.read()
        if not ok: break
        out.write(cv2.flip(fr, 1))
    cap.release(); out.release()
    return output_mp4

# ======================= Multi-signal direction detection (knees, ankles, wrists, hip/shoulder center, nose + optic flow) =======================
def _median_delta(arr):
    """Robust net displacement between early and late segments."""
    if not arr or len(arr) < 4:
        return 0.0, 0.0
    q = max(1, len(arr)//4)
    early = float(np.nanmedian(arr[:q]))
    late  = float(np.nanmedian(arr[-q:]))
    span  = float(np.nanstd(arr) + 1e-6)
    return (late - early), span

def _gather_pose_xsignals(lms):
    """Collect normalized x positions (0..1) from many landmarks."""
    PL = POSE.PoseLandmark
    def ok(idx, thr=0.25):
        lm = lms[idx]; vis = float(getattr(lm, "visibility", 0.0))
        return (lm.x, lm.y) if vis >= thr else None

    xs = {}
    LK = ok(PL.LEFT_KNEE.value);     RK = ok(PL.RIGHT_KNEE.value)
    LA = ok(PL.LEFT_ANKLE.value);    RA = ok(PL.RIGHT_ANKLE.value)
    LW = ok(PL.LEFT_WRIST.value);    RW = ok(PL.RIGHT_WRIST.value)
    LH = ok(PL.LEFT_HIP.value);      RH = ok(PL.RIGHT_HIP.value)
    LS = ok(PL.LEFT_SHOULDER.value); RS = ok(PL.RIGHT_SHOULDER.value)
    NO = ok(PL.NOSE.value)

    if LK: xs.setdefault("knees_L", []).append(LK[0])
    if RK: xs.setdefault("knees_R", []).append(RK[0])
    if LA: xs.setdefault("ankles_L", []).append(LA[0])
    if RA: xs.setdefault("ankles_R", []).append(RA[0])
    if LW: xs.setdefault("wrists_L", []).append(LW[0])
    if RW: xs.setdefault("wrists_R", []).append(RW[0])
    if LH and RH: xs.setdefault("hip_center", []).append((LH[0]+RH[0])/2.0)
    if LS and RS: xs.setdefault("shoulder_center", []).append((LS[0]+RS[0])/2.0)
    if NO: xs.setdefault("nose", []).append(NO[0])

    pts = [p for p in (LK, RK, LA, RA, LW, RW, LH, RH, LS, RS, NO) if p]
    return xs, pts

def detect_run_direction_multi(video_path: str, sample_seconds: float = 3.0) -> tuple[str, float]:
    """
    Fuse multiple pose streams + optic flow to estimate motion direction.
    Returns ('left'|'right'|'unknown', confidence[0..1]).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "unknown", 0.0
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*sample_seconds)
    maxN = int(min(tot, fps*sample_seconds))
    step = max(1, int(fps // 4))  # ~4 fps sampling
    idxs = list(range(0, maxN, step))

    series = {
        "knees_L":[], "knees_R":[], "ankles_L":[], "ankles_R":[], "wrists_L":[], "wrists_R":[], 
        "hip_center":[], "shoulder_center":[], "nose":[]
    }

    flow_x = []
    prev_gray = None

    with POSE.Pose(static_image_mode=False, model_complexity=1,
                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for i in idxs:
            if tot > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None: continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            bbox = None
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                xs, pts = _gather_pose_xsignals(lms)
                for k, arr in xs.items():
                    series[k].extend(arr)
                if pts:
                    xs_pt = [p[0]*w for p in pts]
                    ys_pt = [p[1]*h for p in pts]
                    x1, y1 = int(max(0, min(xs_pt))), int(max(0, min(ys_pt)))
                    x2, y2 = int(min(w-1, max(xs_pt))), int(min(h-1, max(ys_pt)))
                    pad = int(0.12 * max(1, x2-x1))
                    x1 = max(0, x1 - pad); x2 = min(w-1, x2 + pad)
                    y1 = max(0, y1 - pad); y2 = min(h-1, y2 + pad)
                    if x2 > x1 and y2 > y1:
                        bbox = (x1, y1, x2, y2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                if bbox:
                    x1, y1, x2, y2 = bbox
                    roi_prev = prev_gray[y1:y2, x1:x2]
                    roi_now  = gray[y1:y2,  x1:x2]
                else:
                    roi_prev = prev_gray
                    roi_now  = gray
                if roi_prev.size > 0 and roi_now.size > 0 and roi_prev.shape == roi_now.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        roi_prev, roi_now, None,
                        pyr_scale=0.5, levels=3, winsize=21,
                        iterations=3, poly_n=5, poly_sigma=1.1, flags=0
                    )
                    fx_med = float(np.median(flow[...,0]))
                    flow_x.append(fx_med)
            prev_gray = gray

    cap.release()

    votes = []  # (side, conf, weight)
    weights = {
        "knees_L":1.0, "knees_R":1.0, "ankles_L":0.8, "ankles_R":0.8,
        "wrists_L":0.6, "wrists_R":0.6, "hip_center":1.0, "shoulder_center":0.9, "nose":0.5,
        "optic_flow":1.2
    }

    def add_vote(name, arr):
        dx, span = _median_delta(arr)
        if span <= 1e-6: return
        z = dx / (span*2.5)
        conf = float(min(1.0, max(0.0, abs(z))))
        if dx > 0:  votes.append(("right", conf, weights.get(name,1.0)))
        elif dx < 0: votes.append(("left",  conf, weights.get(name,1.0)))

    for key, arr in series.items():
        if len(arr) >= 4:
            add_vote(key, arr)

    if len(flow_x) >= 4:
        span = float(np.std(flow_x) + 1e-6)
        z = (float(np.median(flow_x[-max(1,len(flow_x)//4):])) - float(np.median(flow_x[:max(1,len(flow_x)//4)]))) / (max(span,1e-3)*2.5)
        conf = float(min(1.0, max(0.0, abs(z))))
        if z > 0:  votes.append(("right", conf, weights["optic_flow"]))
        elif z < 0: votes.append(("left",  conf, weights["optic_flow"]))

    if not votes:
        return "unknown", 0.0

    W_left  = sum(w for s,c,w in votes if s == "left")
    W_right = sum(w for s,c,w in votes if s == "right")
    side = "left" if W_left > W_right else ("right" if W_right > W_left else "unknown")
    total_w = W_left + W_right
    agree = (max(W_left, W_right) / total_w) if total_w > 0 else 0.0
    if side != "unknown":
        mean_conf = np.mean([c for s,c,w in votes if s == side]) if any(s==side for s,_,_ in votes) else 0.0
        conf = float(max(0.0, min(1.0, 0.5*agree + 0.5*mean_conf)))
    else:
        conf = 0.0
    return side, conf


def auto_flip_for_direction(inp_path: Path, expected: str = "left") -> tuple[Path, str]:
    """
    Detect screen motion via multi-signal fusion, account for mirrored scenes,
    and flip if opposite to expected ('left' or 'right').
    """
    direction, conf = detect_run_direction_multi(str(inp_path), sample_seconds=3.0)
    msg = []

    # If the scene is mirrored, invert our label so it matches human L/R perception
    mirror_note = ""
    try:
        if direction in ("left", "right") and estimate_scene_mirrored(str(inp_path)):
            direction = "right" if direction == "left" else "left"
            mirror_note = " (mirrored scene detected; interpreting opposite)"
    except Exception:
        pass  # if mirror check fails, continue with original label

    if direction == "unknown" or conf < 0.40:
        msg.append(f"Direction inconclusive (conf {conf*100:.0f}%).")
        return inp_path, " ".join(msg)

    need_flip = (expected == "left" and direction == "right") or (expected == "right" and direction == "left")

    if need_flip:
        flipped_path = inp_path.with_name(inp_path.stem + "_flipped.mp4")
        flip_transcode_cv2(inp_path, flipped_path)
        try:
            os.replace(flipped_path, inp_path)  # keep downstream paths unchanged
        except Exception:
            inp_path = flipped_path
        msg.append(f"Auto-flipped horizontally ({direction}→{expected}, conf {conf*100:.0f}%).{mirror_note}")
        return inp_path, " ".join(msg)

    msg.append(f"Direction OK ({direction}, conf {conf*100:.0f}%).{mirror_note}")
    return inp_path, " ".join(msg)

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def ensure_state():
    if "xstep_out" not in st.session_state:
        st.session_state["xstep_out"] = None

ensure_state()

# ======================= Dim UI =======================
from html import escape
def dim_tab_ui(message: str):
    safe_msg = escape(message)
    st.markdown(
        f"""
        <style>
          div[data-baseweb="tab-list"] button[role="tab"]:nth-child(2) {{
            opacity: .5 !important;
            cursor: not-allowed !important;
          }}
          .xstep-adv-dim {{
            opacity: .5;
            filter: grayscale(20%);
            pointer-events: none;
            transition: opacity .25s ease;
            position: relative;
            min-height: 220px;
          }}
          .xstep-adv-overlay {{
            position: absolute; inset: 0;
            display: grid; place-items: center;
            font-weight: 700; text-align: center; padding: 0 16px;
            font-family:
              "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans",
              "Droid Sans", "Helvetica Neue", Arial,
              "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
          }}
        </style>
        <div class="xstep-adv-dim">
          <div class="xstep-adv-overlay">{safe_msg}</div>
        """,
        unsafe_allow_html=True,
    )
def end_dim_advanced_tab_ui():
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= UI =======================
st.title("⚡️ – Xstep – Running Posture Analysis")
st.caption("Upload an .mp4. The app will normalize orientation & direction, then analyze.")

tab_upload, tab_adv = st.tabs(["📤 Upload Video", "📈 Advanced Analytics"])

# ======================= Upload & Results =======================
with tab_upload:
    uploaded = st.file_uploader("Upload your .mp4 clip", type=["mp4"], key="uploader")

    file_hash = None
    file_bytes = None
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_hash = sha256_bytes(file_bytes)

    need_process = False
    if uploaded is not None:
        if (st.session_state["xstep_out"] is None) or (st.session_state["xstep_out"].get("input_hash") != file_hash):
            need_process = True

    if uploaded is not None and need_process:
        with st.spinner("Processing video..."):
            work = Path(tempfile.mkdtemp(prefix="xstep_"))
            inp = work / "input.mp4"; out_dir = work / "out"
            out_dir.mkdir(exist_ok=True)
            with open(inp, "wb") as f:
                f.write(file_bytes)

            # 1) auto-rotate to upright
            inp_upright, rot_msg = auto_upright_video(inp)

            # 2) NEW: knee-based direction detection & flip (expected training direction = 'left')
            inp_dir_norm, dir_msg = auto_flip_for_direction(inp_upright, expected="left")

            # 3) run core pipeline on normalized video
            xstep.process_video(str(inp_dir_norm), str(out_dir))

            # --- Fallback: if no running detected, try flipping and re-run once ---
            summaries_tmp = load_all_summaries(out_dir)
            running_ok_tmp = not no_running_via_recommendation(summaries_tmp) if summaries_tmp else False
            if not running_ok_tmp:
                # flip regardless of prior decision and re-run
                force_flip_path = Path(str(inp_dir_norm).replace(".mp4", "_forceflip.mp4"))
                flip_transcode_cv2(inp_dir_norm, force_flip_path)
                xstep.process_video(str(force_flip_path), str(out_dir))

            # gather
            summaries = load_all_summaries(out_dir)
            recs = []
            running_ok = False
            posture_majority = "-"
            posture_pct = {"Straight":0.0,"Curved forward":0.0,"Arched (backward)":0.0}
            scores_0_10 = zeros_scores()

            if summaries:
                running_ok = not no_running_via_recommendation(summaries)
                posture_majority, posture_pct = derive_posture_label_stats(summaries)
                avg_raw, recs = aggregate_numeric_scores(summaries)
                scores_0_10 = norm_to_10(avg_raw) if running_ok else zeros_scores()

            status_msgs = []
            if rot_msg: status_msgs.append(rot_msg)
            if dir_msg: status_msgs.append(dir_msg)
            if summaries and (not running_ok):
                status_msgs.append("No running detected in this video.")

            fig = make_radar(scores_0_10)
            assets = Path("assets"); assets.mkdir(exist_ok=True)
            fig_png = assets / f"radar_{int(time.time())}.png"
            try:
                fig.write_image(str(fig_png))
            except Exception:
                fig_png = Path("")

            ann = out_dir / "annotated.mp4"
            disp_bytes = None
            thumb = assets / f"thumb_{int(time.time())}.jpg"
            if ann.exists():
                disp_path = reencode_h264(ann, ann.with_name(ann.stem + "_h264.mp4"))
                with open(disp_path, "rb") as vf:
                    disp_bytes = vf.read()
                extract_video_thumbnail(str(ann), thumb)

            # Build PDF (best-effort)
            pdf_bytes = None
            if running_ok:
                try:
                    pdf_bytes = build_pdf(scores_0_10, fig_png, thumb if thumb.exists() else Path(""),
                                          recs, posture_majority, posture_pct)
                except Exception:
                    pdf_bytes = None

            st.session_state["xstep_out"] = {
                "input_hash": file_hash,
                "rot_msg": " ".join(status_msgs),
                "out_dir": str(out_dir),
                "summaries": summaries,
                "posture_majority": posture_majority,
                "posture_pct": posture_pct,
                "scores_0_10": scores_0_10,
                "running_ok": bool(running_ok),
                "recs": recs,
                "fig_png": str(fig_png) if fig_png and fig_png.exists() else "",
                "thumb": str(thumb) if thumb and thumb.exists() else "",
                "disp_video_bytes": disp_bytes,
                "pdf_bytes": pdf_bytes,
            }

    # Render cached
    X = st.session_state.get("xstep_out")
    if uploaded is not None and X:
        msg = X.get("rot_msg") or ""
        if "Auto-rotating" in msg:
            st.info(msg)
        elif "low confidence" in msg.lower():
            st.warning(msg)
        else:
            st.success(msg or "Orientation & direction OK.")

        if not X.get("running_ok"):
            st.error("No running detected in this video.")

        st.subheader("Posture")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric("Majority", X["posture_majority"])
        with colB:
            st.write("**Distribution**")
            for k, v in X["posture_pct"].items():
                st.progress(v/100.0, text=f"{k}: {v}%")

        st.subheader("Scores by section")
        for k in CATS_NUMERIC:
            val = X["scores_0_10"].get(k, 0.0)
            st.write(f"**{CAT_LABEL[k]}**: {val}/10")
            st.progress(min(max(val/10.0,0),1))

        st.subheader("Weak areas chart")
        st.plotly_chart(make_radar(X["scores_0_10"]), use_container_width=True)

        st.subheader("Annotated video")
        if X.get("disp_video_bytes"):
            render_video_compact(X["disp_video_bytes"], width_px=VIDEO_W)
        else:
            st.info("No annotated video found.")

        if X.get("recs"):
            st.subheader("Recommendations")
            for r in X["recs"]:
                st.write(f"- {r}")

        st.subheader("Export")
        if X["running_ok"]:
            if X.get("pdf_bytes"):
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=X["pdf_bytes"],
                    file_name="Xstep_Posture_Report.pdf",
                    mime="application/pdf",
                    key="dl_pdf"
                )
            else:
                st.error("PDF export failed earlier. Re-upload to try again.")
        else:
            st.info("No running/jogging segments detected. PDF export is disabled.")
        

# NOT WORKING
# ======================= PDF builder =======================
def build_pdf(scores_0_10: dict, radar_png: Path, thumb_jpg: Path,
              recommendations: list[str], posture_majority: str, posture_pct: dict):

    # Create a compact A4 PDF summary. Safe if images are missing.
    # Returns bytes suitable for Streamlit's download_button.

    try:
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=12)

        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 10, "Xstep – Running Posture Report", ln=1)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, "Auto-normalized orientation & direction; scored per-lap.")

        # Posture summary
        pdf.ln(2); pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, "Posture Summary", ln=1)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6, f"Majority posture: {posture_majority}", ln=1)
        if posture_pct:
            s_straight = posture_pct.get("Straight", 0)
            s_curved   = posture_pct.get("Curved forward", 0)
            s_arched   = posture_pct.get("Arched (backward)", 0)
            pdf.cell(0, 6, f"Straight: {s_straight}% | Curved forward: {s_curved}% | Arched: {s_arched}%", ln=1)

        # Scores
        pdf.ln(1); pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, "Scores (0–10)", ln=1)
        pdf.set_font("Arial", "", 11)
        for key in CATS_NUMERIC:
            label = CAT_LABEL.get(key, key)
            val = scores_0_10.get(key, 0.0)
            pdf.cell(0, 6, f"{label}: {val}/10", ln=1)

        # Images row
        placed_any = False
        y_start = pdf.get_y()
        if thumb_jpg and str(thumb_jpg) and Path(thumb_jpg).exists():
            try:
                pdf.ln(2); pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, "Annotated Frame", ln=1)
                x_thumb = pdf.get_x(); y_thumb = pdf.get_y()
                pdf.image(str(thumb_jpg), w=90)
                placed_any = True
            except Exception:
                pass

        if radar_png and str(radar_png) and Path(radar_png).exists():
            try:
                if placed_any:
                    # place radar to the right of the thumb
                    pdf.set_xy(110, y_thumb)
                else:
                    pdf.ln(2); pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, "Weak Areas Chart", ln=1)
                pdf.image(str(radar_png), w=85)
                placed_any = True
            except Exception:
                pass

        # Recommendations
        if recommendations:
            pdf.ln(4 if placed_any else 2)
            pdf.set_font("Arial", "B", 13); pdf.cell(0, 8, "Recommendations", ln=1)
            pdf.set_font("Arial", "", 11)
            for r in recommendations:
                pdf.multi_cell(0, 6, f"- {r}")

        # Return bytes
        out = pdf.output(dest="S")
        if isinstance(out, bytes):
            return out
        return out.encode("latin-1")
    except Exception:
        # Keep the app running even if PDF fails
        return None

# ======================= Advanced Analytics =======================
with tab_adv:
    X = st.session_state.get("xstep_out")

    if not X:
        dim_tab_ui("❌ Advanced Analytics is Disabled (Upload a video first).")
        end_dim_advanced_tab_ui()
        st.stop()

    if not X.get("running_ok"):
        dim_tab_ui("❌ Advanced Analytics is Disabled (No running/jogging video detected).")
        end_dim_advanced_tab_ui()
        st.stop()

    out_dir = Path(X["out_dir"])
    posture_majority = X["posture_majority"]
    posture_pct = X["posture_pct"]

    st.markdown("### Posture overview")
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Majority posture", posture_majority)
    with col2:
        st.write("**Distribution**")
        for k, v in posture_pct.items():
            st.progress(v/100.0, text=f"{k}: {v}%")

    df = read_features_csv(out_dir)
    if df is None or df.empty:
        st.info("No per-frame CSV found. (Expected features.csv)")
        st.stop()

    if "frame" not in df.columns:
        df.insert(0, "frame", range(len(df)))
    df = ensure_numeric(df, ["trunk_lean_deg","L_shank_deg","R_shank_deg","L_foot_pitch_deg","R_foot_pitch_deg"])

    st.markdown("### Posture timeline")
    if "posture_label" in df.columns and df["posture_label"].notna().any():
        order = ["Straight", "Curved forward", "Arched (backward)"]
        cat = pd.Categorical(df["posture_label"], categories=order, ordered=True)
        df_plot = df.assign(posture_label_cat=cat)
        fig_tl = px.scatter(
            df_plot.dropna(subset=["posture_label_cat"]),
            x="frame", y="posture_label_cat", color="posture_label_cat",
            title="Per-frame posture label", height=320
        )
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("No per-frame posture labels found in CSV.")

    st.markdown("### L/R Symmetry timelines")
    if {"L_shank_deg","R_shank_deg"}.issubset(df.columns):
        df_sh = df[["frame","L_shank_deg","R_shank_deg"]].dropna(how="all", subset=["L_shank_deg","R_shank_deg"])
        if not df_sh.empty:
            fig_sh = px.line(df_sh, x="frame", y=["L_shank_deg","R_shank_deg"],
                             labels={"value":"Shank angle (°)"}, title="Shank angles (L/R) over frames", height=320)
            st.plotly_chart(fig_sh, use_container_width=True)

            msa_L = running_nanmax(df_sh["L_shank_deg"])
            msa_R = running_nanmax(df_sh["R_shank_deg"])
            df_msa = pd.DataFrame({"frame": df_sh["frame"], "MSA_L_running_max": msa_L, "MSA_R_running_max": msa_R})
            fig_msa = px.line(df_msa, x="frame", y=["MSA_L_running_max","MSA_R_running_max"],
                              labels={"value":"Running max shank (°)"}, title="MSA running max (L/R) over frames", height=320)
            st.plotly_chart(fig_msa, use_container_width=True)

            df_diff = pd.DataFrame({
                "frame": df_sh["frame"],
                "abs_diff_deg": np.abs(pd.to_numeric(df_sh["L_shank_deg"], errors="coerce") -
                                       pd.to_numeric(df_sh["R_shank_deg"], errors="coerce"))
            })
            fig_diff = px.line(df_diff, x="frame", y="abs_diff_deg",
                               title="Instant L/R shank imbalance |L-R| (°)", height=280)
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("Shank columns are present, but all values are empty.")
    else:
        st.info("No L/R shank columns found for symmetry timelines.")

    st.markdown("### Other timelines")
    if "trunk_lean_deg" in df.columns and df["trunk_lean_deg"].notna().any():
        fig_lean = px.line(df, x="frame", y="trunk_lean_deg",
                           labels={"trunk_lean_deg":"Trunk lean (°)"},
                           title="Trunk lean over frames", height=300)
        st.plotly_chart(fig_lean, use_container_width=True)

    fp_cols = [c for c in ["L_foot_pitch_deg","R_foot_pitch_deg"] if c in df.columns and df[c].notna().any()]
    if fp_cols:
        fig_fp = px.line(df, x="frame", y=fp_cols, title="Foot pitch (L/R) over frames", height=300)
        st.plotly_chart(fig_fp, use_container_width=True)

    def plot_strike(side: str):
        col = f"{side}_strike"; nice = "Left" if side=="L" else "Right"
        if col in df.columns and df[col].astype(str).str.len().gt(0).any():
            order = ["heel","midfoot","forefoot"]
            ymap = {k:i for i,k in enumerate(order)}
            d = df[["frame", col]].copy()
            d["strike_idx"] = d[col].map(ymap)
            d = d.dropna(subset=["strike_idx"])
            if not d.empty:
                fig = px.scatter(d, x="frame", y="strike_idx", color=col,
                                 category_orders={col: order},
                                 title=f"{nice} foot strike over frames", height=260)
                fig.update_yaxes(tickmode="array", tickvals=[0,1,2], ticktext=order, title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {nice.lower()} foot strike events to plot.")
        else:
            st.info(f"No {nice.lower()} foot strike column in CSV.")

    plot_strike("L"); plot_strike("R")

    if "activity" in df.columns and df["activity"].astype(str).str.len().gt(0).any():
        order = ["not_active","walking","jogging","running"]
        ymap = {k:i for i,k in enumerate(order)}
        d = df[["frame","activity"]].copy()
        d["act_idx"] = d["activity"].map(ymap)
        d = d.dropna(subset=["act_idx"])
        if not d.empty:
            fig_act = px.scatter(d, x="frame", y="act_idx", color="activity",
                                 category_orders={"activity": order},
                                 title="Activity over frames", height=260)
            fig_act.update_yaxes(tickmode="array", tickvals=[0,1,2,3], ticktext=order, title=None)
            st.plotly_chart(fig_act, use_container_width=True)
        else:
            st.info("No activity rows to plot.")
    else:
        st.info("No activity column in CSV.")
