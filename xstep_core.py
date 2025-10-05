"""
Xstep_mediapipe.py
Running Posture Correction Analysis Script (debounced + min-lap-duration + robust realtime I/O + live summary)
- Anti-spike activity debounce so it doesn't flash "running" at start.
- Ignores any running/jogging segment shorter than 3 seconds (no summary).
- Realtime mode: robust camera open on Windows, MP4/AVI writer fallbacks, CSV flush.
- Live summary: writes a rolling JSON file (live_summary.json) in realtime mode, now with recommendations.

"""
import os, sys, csv, json, platform, time
import cv2
import numpy as np
from pathlib import Path

# ---------- Model loader helper ----------
import joblib, zipfile, requests
from pathlib import Path

def _load_activity_model():
    """
    Try to load the activity model from multiple sources:
      1. Local 'models/activity_rf.joblib' or 'activity_rf.zip'
      2. Cached file under '.cache_models/activity_rf.joblib'
      3. Download via URL in Streamlit secrets or env var 'MODEL_URL'
    """
    base = Path(__file__).resolve().parent
    model_dir = base / "models"
    local_files = [
        model_dir / "activity_rf.joblib",
        model_dir / "activity_rf.zip",
    ]
    for path in local_files:
        if path.exists():
            if path.suffix == ".zip":
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(model_dir)
                    inner = model_dir / "activity_rf.joblib"
                    if inner.exists():
                        return joblib.load(inner)
            else:
                return joblib.load(path)

    # --- fallback: download ---
    url = os.getenv("MODEL_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("MODEL_URL")
        except Exception:
            url = None
    if not url:
        print("[_load_activity_model] No local or remote model found.")
        return None

    cache_dir = base / ".cache_models"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "activity_rf.joblib"

    if not cache_file.exists():
        print(f"[_load_activity_model] Downloading model from {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if url.endswith(".zip"):
            ztmp = cache_dir / "activity_rf.zip"
            ztmp.write_bytes(r.content)
            with zipfile.ZipFile(ztmp, "r") as zf:
                zf.extractall(cache_dir)
            if cache_file.exists():
                return joblib.load(cache_file)
        else:
            cache_file.write_bytes(r.content)
    return joblib.load(cache_file) if cache_file.exists() else None

# Try to import joblib for loading model (if available)
try:
    import joblib
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

SAVE_FLAGGED_FRAMES = False

# MediaPipe pose landmark indices (33 landmarks)
POSE_LM = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# ---------- Geometry helpers ----------
def angle_deg(v1, v2):
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    c = np.dot(v1, v2) / (n1 * n2)
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def line_angle_with_vertical(p_top, p_bottom):
    v = np.array(p_top, float) - np.array(p_bottom, float)
    return angle_deg(v, np.array([0, -1], float))

def foot_pitch_deg(heel, toe):
    v = np.array(toe, float) - np.array(heel, float)
    return angle_deg(v, np.array([1, 0], float))

def shank_angle_deg(knee, ankle):
    return line_angle_with_vertical(knee, ankle)

def trunk_lean_deg(hip_center, shoulder_center):
    ang = line_angle_with_vertical(shoulder_center, hip_center)
    if ang is None:
        return None
    v = np.array(shoulder_center, float) - np.array(hip_center, float)
    sign = 1 if v[1] > 0 else -1  # image y increases downward
    return ang * sign

def midpoint(a, b):
    return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0) if a is not None and b is not None else None

# ---------- Filters ----------
class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        if x is None:
            return self.v
        self.v = x if self.v is None else (1 - self.alpha) * self.v + self.alpha * x
        return self.v

class Velocity:
    def __init__(self, maxlen=3):
        from collections import deque
        self.buf = deque(maxlen=maxlen)
    def update(self, x):
        self.buf.append(x)
        if len(self.buf) < 2:
            return 0.0
        return self.buf[-1] - self.buf[-2]

# ---------- Foot strike & posture labels ----------
def classify_foot_strike(ankle_y, heel_y, toe_y, vy_ankle, y_ground_threshold=0):
    contact = (vy_ankle >= -0.5) and (ankle_y is not None and ankle_y > y_ground_threshold)
    if not contact:
        return None
    if heel_y - toe_y > 5:
        return "heel"
    if abs(heel_y - toe_y) <= 5:
        return "midfoot"
    return "forefoot"

def label_trunk_posture(lean_deg, straight_thr=6, arched_thr=-6):
    if lean_deg is None:
        return None
    if lean_deg > straight_thr:
        return "Curved forward"
    if lean_deg < arched_thr:
        return "Arched (backward)"
    return "Straight"

# ---------- Feature extraction ----------
def extract_running_features(landmarks, image_w, image_h, prev_state):
    def safe_pt(lm, conf_thr=0.4):
        return None if lm is None or lm[2] < conf_thr else (lm[0], lm[1])

    l_sh = safe_pt(landmarks.get('left_shoulder')); r_sh = safe_pt(landmarks.get('right_shoulder'))
    l_hip = safe_pt(landmarks.get('left_hip'));      r_hip = safe_pt(landmarks.get('right_hip'))
    l_knee = safe_pt(landmarks.get('left_knee'));    r_knee = safe_pt(landmarks.get('right_knee'))
    l_ank = safe_pt(landmarks.get('left_ankle'));    r_ank = safe_pt(landmarks.get('right_ankle'))
    l_heel = safe_pt(landmarks.get('left_heel'));    r_heel = safe_pt(landmarks.get('right_heel'))
    l_toe = safe_pt(landmarks.get('left_foot_index')); r_toe = safe_pt(landmarks.get('right_foot_index'))

    shoulder_center = midpoint(l_sh, r_sh)
    hip_center = midpoint(l_hip, r_hip)
    trunk_lean = trunk_lean_deg(hip_center, shoulder_center) if (hip_center and shoulder_center) else None
    L_shank = shank_angle_deg(l_knee, l_ank) if (l_knee and l_ank) else None
    R_shank = shank_angle_deg(r_knee, r_ank) if (r_knee and r_ank) else None
    L_foot_pitch = foot_pitch_deg(l_heel, l_toe) if (l_heel and l_toe) else None
    R_foot_pitch = foot_pitch_deg(r_heel, r_toe) if (r_heel and r_toe) else None

    if 'MSA_L' not in prev_state:
        prev_state['MSA_L'] = 0.0
        prev_state['MSA_R'] = 0.0
        prev_state['ema_ankle_y_L'] = EMA(alpha=0.3)
        prev_state['ema_ankle_y_R'] = EMA(alpha=0.3)
        prev_state['vy_ankle_L'] = Velocity()
        prev_state['vy_ankle_R'] = Velocity()

    prev_state['MSA_L'] = max(prev_state.get('MSA_L', 0.0), L_shank or 0.0)
    prev_state['MSA_R'] = max(prev_state.get('MSA_R', 0.0), R_shank or 0.0)

    ank_y_L = prev_state['ema_ankle_y_L'].update(l_ank[1] if l_ank else None)
    ank_y_R = prev_state['ema_ankle_y_R'].update(r_ank[1] if r_ank else None)
    vy_L = prev_state['vy_ankle_L'].update(ank_y_L)
    vy_R = prev_state['vy_ankle_R'].update(ank_y_R)

    L_strike = classify_foot_strike(l_ank[1] if l_ank else None,
                                    l_heel[1] if l_heel else 0,
                                    l_toe[1] if l_toe else 0,
                                    vy_L)
    R_strike = classify_foot_strike(r_ank[1] if r_ank else None,
                                    r_heel[1] if r_heel else 0,
                                    r_toe[1] if r_toe else 0,
                                    vy_R)

    posture_label = label_trunk_posture(trunk_lean)
    features = {
        'posture_label': posture_label or "",
        'trunk_lean_deg': trunk_lean,
        'L_shank_deg': L_shank,
        'R_shank_deg': R_shank,
        'L_foot_pitch_deg': L_foot_pitch,
        'R_foot_pitch_deg': R_foot_pitch,
        'L_strike': L_strike or "",
        'R_strike': R_strike or "",
        'MSA_L_running_max': prev_state['MSA_L'],
        'MSA_R_running_max': prev_state['MSA_R']
    }
    return features, prev_state

# ---------- Activity detector with debounce ----------
class RunningDetector:
    """
    Detects running/jogging vs walking/not_active using a trained model (if available) or a heuristic.
    Adds DEBOUNCE so classification labels don't spike (e.g., show "running" at video start).
    """
    def __init__(self, fps, window_sec=6.0,
                 anti_spike_to_run_sec=0.75,   # require this long to promote -> jogging/running
                 anti_spike_to_idle_sec=0.50,  # require this long to demote -> walking/not_active
                 anti_spike_between_run_sec=0.30):  # jogging<->running smoothing
        from collections import deque
        self.fps = fps
        self.window = int(max(1, round(window_sec * fps)))
        self.frame_idx = 0

        # Model buffer
        self.model = None
        self.model_labels = []
        self.model_window = int(round(2.0 * fps))
        self.recent_frames = deque(maxlen=self.model_window)

        # Heuristic buffers
        self.left_strikes = deque()
        self.right_strikes = deque()

        # Debounce
        self.stable_state = "not_active"
        self.candidate_state = None
        self.candidate_count = 0

        self.to_run_frames  = max(1, int(round(anti_spike_to_run_sec * fps)))
        self.to_idle_frames = max(1, int(round(anti_spike_to_idle_sec * fps)))
        self.between_run_frames = max(1, int(round(anti_spike_between_run_sec * fps)))

       if HAVE_JOBLIB:
    try:
        md = _load_activity_model()
        if isinstance(md, dict):
            self.model = md.get("model", None)
            self.model_labels = md.get("labels", [])
        else:
            self.model = md
            self.model_labels = []
        if self.model is None:
            print("[RunningDetector] Warning: Model object missing, using heuristic.")
    except Exception as e:
        print(f"[RunningDetector] Warning: cannot load model -> heuristic (reason: {e})")
        self.model = None
else:
    print("[RunningDetector] joblib not available -> heuristic fallback.")

    def _trim(self, deq):
        while deq and (self.frame_idx - deq[0] > self.window):
            deq.popleft()

    def _hz(self, deq):
        if len(deq) < 2:
            return 0.0
        duration = deq[-1] - deq[0]
        if duration <= 0:
            return 0.0
        steps = len(deq) - 1
        return steps / (duration / self.fps)

    def _model_features_from_window(self):
        recent = list(self.recent_frames)
        def agg(key, keep_range=False):
            vals = [f.get(key) for f in recent if f.get(key) is not None]
            if not vals:
                return [0.0, 0.0, 0.0] if keep_range else [0.0, 0.0]
            arr = np.asarray(vals, float)
            mean = float(np.nanmean(arr)); std = float(np.nanstd(arr))
            if keep_range:
                rng = float(np.nanmax(arr) - np.nanmin(arr))
                return [mean, std, rng]
            return [mean, std]

        feats = []
        feats += agg("trunk_lean_deg", keep_range=True)
        feats += agg("L_shank_deg") + agg("R_shank_deg")
        feats += agg("L_foot_pitch_deg") + agg("R_foot_pitch_deg")
        for side in ["L_strike", "R_strike"]:
            vals = [str(f.get(side) or "") for f in recent]
            feats += [vals.count("heel"), vals.count("midfoot"), vals.count("forefoot")]
            prev = None; trans = 0
            for s in vals:
                if s and prev is not None and s != prev:
                    trans += 1
                if s:
                    prev = s
            feats += [trans]
        return np.array([feats], dtype=np.float32)

    def _debounce(self, new_state):
        def need_frames(from_state, to_state):
            if to_state in ("running", "jogging") and from_state in ("not_active", "walking"):
                return self.to_run_frames
            if to_state in ("not_active", "walking") and from_state in ("running", "jogging"):
                return self.to_idle_frames
            if (to_state in ("running", "jogging")) and (from_state in ("running", "jogging")):
                return self.between_run_frames
            if (to_state in ("not_active", "walking")) and (from_state in ("not_active", "walking")):
                return 1
            return max(1, self.between_run_frames // 2)

        if new_state == self.stable_state:
            self.candidate_state = None
            self.candidate_count = 0
            return self.stable_state

        if self.candidate_state != new_state:
            self.candidate_state = new_state
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= need_frames(self.stable_state, new_state):
            self.stable_state = new_state
            self.candidate_state = None
            self.candidate_count = 0

        return self.stable_state

    def update(self, features):
        # Record foot strike events (for heuristic)
        if features.get('L_strike'):
            self.left_strikes.append(self.frame_idx)
        if features.get('R_strike'):
            self.right_strikes.append(self.frame_idx)
        self._trim(self.left_strikes); self._trim(self.right_strikes)

        hz_L = self._hz(self.left_strikes)
        hz_R = self._hz(self.right_strikes)
        spm_total = 60.0 * (hz_L + hz_R)

        # 1) Raw activity_state from model (if available and warmed) else heuristic
        activity_state = None
        if self.model is not None and len(self.recent_frames) >= self.model_window:
            try:
                X = self._model_features_from_window()
                pred_idx = int(self.model.predict(X)[0])
                activity_state = self.model_labels[pred_idx] if self.model_labels else str(pred_idx)
            except Exception as e:
                print(f"[RunningDetector] model prediction failed -> fallback (reason: {e})")
                self.model = None
                activity_state = None

        if activity_state is None:
            # Heuristic tiers by steps-per-minute
            if (len(self.left_strikes) + len(self.right_strikes)) < 3:
                activity_state = "not_active"
            else:
                if spm_total < 80:
                    activity_state = "not_active"
                elif spm_total < 120:
                    activity_state = "walking"
                elif spm_total < 150:
                    activity_state = "jogging"
                else:
                    activity_state = "running"

        # 2) Debounce (anti-spike)
        debounced_state = self._debounce(activity_state)

        self.recent_frames.append(features)
        self.frame_idx += 1
        return {"state": debounced_state, "spm_total": spm_total, "hz_L": hz_L, "hz_R": hz_R}

# ---------- Per-lap stance scorer ----------
class StanceScorer:
    def __init__(self, fps):
        from collections import defaultdict
        self.fps = fps
        self.tally = defaultdict(float)
        self.counts = defaultdict(int)
        self.details = []
        self.msa_L_max = 0.0
        self.msa_R_max = 0.0
        self.msa_seen = False

    @staticmethod
    def _score_posture(lean_deg, activity):
        if lean_deg is None or activity in ("not_active", "walking"):
            return 0.0, "N/A"
        if -6 <= lean_deg <= 6:
            return +2.0, "Straight"
        if 6 < lean_deg <= 15:
            return +1.0, "Curved forward"
        if lean_deg > 15:
            return -0.5, "Too forward"
        if -15 <= lean_deg < -6:
            return -1.0, "Slight arch"
        return -2.0, "Arched"

    @staticmethod
    def _score_shank(shank_deg, activity):
        if shank_deg is None or activity in ("not_active", "walking"):
            return 0.0
        if 10 <= shank_deg <= 60:
            return +1.0
        if 5 <= shank_deg < 10 or 60 < shank_deg <= 70:
            return +0.3
        return -0.7

    @staticmethod
    def _score_foot_strike(strike, activity):
        if not strike or activity in ("not_active", "walking"):
            return 0.0
        if activity == "jogging":
            if strike == "midfoot":
                return +1.0
            if strike == "forefoot":
                return +0.7
            if strike == "heel":
                return 0.0
        if strike == "midfoot":
            return +1.5
        if strike == "forefoot":
            return +1.0
        if strike == "heel":
            return -0.7
        return 0.0

    @staticmethod
    def _score_foot_pitch_at_contact(pitch_deg, strike, activity):
        if pitch_deg is None or not strike or activity in ("not_active", "walking"):
            return 0.0
        if strike == "midfoot":
            return +0.8 if -5 <= pitch_deg <= 15 else -0.4
        if strike == "forefoot":
            return +0.6 if pitch_deg >= 10 else -0.3
        if strike == "heel":
            return +0.2 if pitch_deg <= -5 else -0.2
        return 0.0

    def _update_msa(self, msa_L, msa_R):
        if msa_L is not None:
            self.msa_L_max = max(self.msa_L_max, msa_L); self.msa_seen = True
        if msa_R is not None:
            self.msa_R_max = max(self.msa_R_max, msa_R); self.msa_seen = True

    def update_frame(self, features, activity_state):
        self._update_msa(features.get('MSA_L_running_max'), features.get('MSA_R_running_max'))

        s_posture, posture_bucket = self._score_posture(features.get('trunk_lean_deg'), activity_state)
        self.tally["posture"] += s_posture; self.counts["posture"] += 1

        s_shank = 0.0
        s_shank += self._score_shank(features.get('L_shank_deg'), activity_state)
        s_shank += self._score_shank(features.get('R_shank_deg'), activity_state)
        self.tally["shank"] += s_shank; self.counts["shank"] += 1

        s_fs = 0.0
        s_fs += self._score_foot_strike(features.get('L_strike'), activity_state)
        s_fs += self._score_foot_strike(features.get('R_strike'), activity_state)
        s_fp = 0.0
        s_fp += self._score_foot_pitch_at_contact(features.get('L_foot_pitch_deg'), features.get('L_strike'), activity_state)
        s_fp += self._score_foot_pitch_at_contact(features.get('R_foot_pitch_deg'), features.get('R_strike'), activity_state)
        self.tally["foot_strike"] += s_fs; self.counts["foot_strike"] += 1
        self.tally["foot_pitch"] += s_fp; self.counts["foot_pitch"] += 1

        self.details.append({
            "activity": activity_state,
            "posture_score": s_posture,
            "posture_bucket": posture_bucket,
            "shank_score": s_shank,
            "foot_strike_score": s_fs,
            "foot_pitch_score": s_fp
        })

    def finalize(self):
        msa_score = 0.0; sym_score = 0.0; notes = {}
        if self.msa_seen:
            for side, val in (("L", self.msa_L_max), ("R", self.msa_R_max)):
                if 40 <= val <= 70:
                    msa_score += 2.0; notes[f"MSA_{side}"] = "good"
                elif 30 <= val < 40 or 70 < val <= 80:
                    msa_score += 0.8; notes[f"MSA_{side}"] = "ok"
                else:
                    msa_score -= 0.6; notes[f"MSA_{side}"] = "out_of_range"
            diff = abs(self.msa_L_max - self.msa_R_max)
            if diff <= 10:
                sym_score += 2.0; notes["symmetry"] = "excellent"
            elif diff <= 20:
                sym_score += 1.0; notes["symmetry"] = "good"
            else:
                sym_score -= 2.0; notes["symmetry"] = "poor"

        self.tally["msa"] += msa_score; self.counts["msa"] += 1
        self.tally["symmetry"] += sym_score; self.counts["symmetry"] += 1

        weights = {"posture": 1.0, "shank": 0.8, "foot_strike": 1.0, "foot_pitch": 0.6, "msa": 1.2, "symmetry": 1.0}
        total_score = 0.0
        segment_weighted = {}
        for segment, raw_score in self.tally.items():
            w = weights.get(segment, 1.0)
            segment_weighted[segment] = raw_score * w
            total_score += segment_weighted[segment]

        report = {
            "segment_raw": dict(self.tally),
            "segment_weighted": segment_weighted,
            "counts": dict(self.counts),
            "session_notes": {**notes, "MSA_L_max": self.msa_L_max, "MSA_R_max": self.msa_R_max},
            "total_score": total_score
        }
        return report

    def snapshot(self):
        """
        Partial, non-final summary suitable for live display (no extra scoring tweaks).
        """
        weights = {"posture": 1.0, "shank": 0.8, "foot_strike": 1.0, "foot_pitch": 0.6}
        segment_weighted = {}
        total_score = 0.0
        for segment in ["posture", "shank", "foot_strike", "foot_pitch"]:
            raw = float(self.tally.get(segment, 0.0))
            w = weights.get(segment, 1.0)
            segment_weighted[segment] = raw * w
            total_score += segment_weighted[segment]
        return {
            "segment_raw_partial": {k: float(self.tally.get(k, 0.0)) for k in ["posture","shank","foot_strike","foot_pitch"]},
            "segment_weighted_partial": segment_weighted,
            "counts": {k: int(self.counts.get(k, 0)) for k in ["posture","shank","foot_strike","foot_pitch"]},
            "MSA_L_max_so_far": float(self.msa_L_max),
            "MSA_R_max_so_far": float(self.msa_R_max),
            "frames_scored": len(self.details),
            "estimated_score_so_far": float(total_score)
        }

# ---------- Shared recommendations helper ----------
def build_recommendations(seg_raw, notes=None, posture_counts=None):
    """
    Generate human-readable recommendations from segment scores and session notes.
    Works for both finalized (batch) and partial (realtime snapshot) data.
    """
    recs = []
    # Core segments
    if seg_raw.get("posture", 0) < 0:
        recs.append("Maintain a more upright torso: avoid leaning too far forward or arching backward while running.")
    if seg_raw.get("shank", 0) < 0:
        recs.append("Avoid overstriding and keep shin angles moderate (roughly 10°–60° at contact).")
    if seg_raw.get("foot_strike", 0) < 0:
        recs.append("Try to land closer to midfoot instead of heel striking for a softer, more efficient landing.")
    if seg_raw.get("foot_pitch", 0) < 0:
        recs.append("Aim to contact the ground with a flatter foot (not excessively toe-pointed or heel-first).")

    # Notes-based (MSA/symmetry) if available
    if notes:
        if notes.get("symmetry") == "poor" or seg_raw.get("symmetry", 0) < 0:
            recs.append("Improve left-right balance with strength and form drills.")

        MSA_L = notes.get("MSA_L_max", 0.0)
        MSA_R = notes.get("MSA_R_max", 0.0)
        if notes.get("MSA_L") == "out_of_range":
            if MSA_L < 40:
                recs.append(f"Increase left knee drive: left maximum shank angle {MSA_L:.1f}° (target ~40–70°).")
            elif MSA_L > 70:
                recs.append(f"Reduce left overstride: left shank angle {MSA_L:.1f}° (above ~40–70°).")
        if notes.get("MSA_R") == "out_of_range":
            if MSA_R < 40:
                recs.append(f"Increase right knee drive: right maximum shank angle {MSA_R:.1f}° (target ~40–70°).")
            elif MSA_R > 70:
                recs.append(f"Reduce right overstride: right shank angle {MSA_R:.1f}° (above ~40–70°).")

    # Friendly default when nothing negative detected
    if not recs:
        if posture_counts and sum(posture_counts.values()) > 0:
            recs.append("Overall running form looks good – no major issues detected.")
        else:
            # for realtime partials with zero frames, keep it quiet; caller can omit
            recs.append("No running/jogging segments detected yet.")
    return recs

# ---------- Batch processing ----------
def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    FPS = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    if W == 0 or H == 0:
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError("Failed to read first frame to infer video size.")
        H, W = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    os.makedirs(output_dir, exist_ok=True)
    vid_out_path = os.path.join(output_dir, "annotated.mp4")
    csv_out_path = os.path.join(output_dir, "features.csv")

    # Clean old lap summaries
    for fname in os.listdir(output_dir):
        if fname.startswith("summary_lap_") and fname.endswith(".json"):
            try:
                os.remove(os.path.join(output_dir, fname))
            except Exception:
                pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(vid_out_path, fourcc, FPS, (W, H))
    csv_file = open(csv_out_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "activity", "posture_label",
        "trunk_lean_deg", "L_shank_deg", "R_shank_deg",
        "L_foot_pitch_deg", "R_foot_pitch_deg", "L_strike", "R_strike"
    ])

    detector = RunningDetector(FPS, window_sec=6.0)
    prev_state = {}

    # Lap segmentation
    lap_active = False
    pending_start = False
    pending_frames = 0
    break_frames = 0

    stance_scorer = None
    posture_counts = None
    strike_counts = None

    confirm_run_frames = int(np.ceil(0.7 * FPS))
    max_break_frames = int(np.ceil(5.0 * FPS))
    min_lap_frames   = int(np.ceil(3.0 * FPS))
    saved_lap_count = 0

    import mediapipe as mp
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                enable_segmentation=False,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            s_post = s_sh = s_fs = s_fp = 0.0
            flagged = False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            lms_dict = {}
            if results.pose_landmarks:
                for name, idx in POSE_LM.items():
                    lm = results.pose_landmarks.landmark[idx]
                    lms_dict[name] = (lm.x * W, lm.y * H, float(getattr(lm, "visibility", 0.0)))

            features, prev_state = extract_running_features(lms_dict, W, H, prev_state)
            act_info = detector.update(features)
            activity_state = act_info["state"]

            # --- Lap logic with start confirmation & min-duration guard ---
            if not lap_active:
                if not pending_start and activity_state in ("running", "jogging"):
                    pending_start = True
                    pending_frames = 1
                elif pending_start:
                    if activity_state in ("running", "jogging"):
                        pending_frames += 1
                        if pending_frames >= confirm_run_frames:
                            lap_active = True
                            pending_start = False
                            pending_frames = 0
                            break_frames = 0
                            stance_scorer = StanceScorer(fps=FPS)
                            posture_counts = {"Straight": 0, "Curved forward": 0, "Arched (backward)": 0}
                            strike_counts = {"L": {"heel": 0, "midfoot": 0, "forefoot": 0},
                                             "R": {"heel": 0, "midfoot": 0, "forefoot": 0}}
                            lap_start_frame_idx = frame_idx
                            lap_running_frames = 0
                            # score the current frame as part of lap
                            stance_scorer.update_frame(features, activity_state)
                            lap_running_frames += 1
                            pb = stance_scorer.details[-1]["posture_bucket"]
                            if pb and pb != "N/A":
                                if pb in ("Straight", "Curved forward", "Arched (backward)"):
                                    posture_counts[pb] += 1
                                elif pb in ("Slight arch", "Arched"):
                                    posture_counts["Arched (backward)"] += 1
                                elif pb == "Too forward":
                                    posture_counts["Curved forward"] += 1
                            Ls = features.get('L_strike'); Rs = features.get('R_strike')
                            if Ls and Ls in strike_counts["L"]:
                                strike_counts["L"][Ls] += 1
                            if Rs and Rs in strike_counts["R"]:
                                strike_counts["R"][Rs] += 1
                    else:
                        pending_start = False
                        pending_frames = 0
            else:
                if activity_state in ("running", "jogging"):
                    break_frames = 0
                    stance_scorer.update_frame(features, activity_state)
                    lap_running_frames += 1

                    s_post = stance_scorer.details[-1]["posture_score"]
                    s_sh = stance_scorer.details[-1]["shank_score"]
                    s_fs = stance_scorer.details[-1]["foot_strike_score"]
                    s_fp = stance_scorer.details[-1]["foot_pitch_score"]
                    flagged = (s_post < 0 or s_sh < 0 or s_fs < 0 or s_fp < 0)

                    pb = stance_scorer.details[-1]["posture_bucket"]
                    if pb and pb != "N/A":
                        if pb in ("Straight", "Curved forward", "Arched (backward)"):
                            posture_counts[pb] += 1
                        elif pb in ("Slight arch", "Arched"):
                            posture_counts["Arched (backward)"] += 1
                        elif pb == "Too forward":
                            posture_counts["Curved forward"] += 1
                    Ls = features.get('L_strike'); Rs = features.get('R_strike')
                    if Ls and Ls in strike_counts["L"]:
                        strike_counts["L"][Ls] += 1
                    if Rs and Rs in strike_counts["R"]:
                        strike_counts["R"][Rs] += 1
                else:
                    break_frames += 1
                    if break_frames > max_break_frames:
                        # End lap; accept only if >= min_lap_frames active
                        lap_end_frame = frame_idx - 1
                        lap_total_frames = lap_end_frame - lap_start_frame_idx + 1
                        if lap_running_frames >= min_lap_frames:
                            saved_lap_count += 1
                            report = stance_scorer.finalize()
                            report["frames"] = lap_total_frames
                            report["active_running_frames"] = lap_running_frames
                            report["posture_distribution"] = posture_counts
                            report["footstrike_distribution"] = strike_counts

                            # Recommendations (shared helper)
                            recs = build_recommendations(report["segment_raw"], report.get("session_notes", {}), posture_counts)
                            report["recommendations"] = recs

                            summary_path = os.path.join(output_dir, f"summary_lap_{saved_lap_count}.json")
                            with open(summary_path, "w", encoding="utf-8") as jf:
                                json.dump(report, jf, indent=2)
                            print(f"Saved lap #{saved_lap_count} (frames={lap_total_frames}, active={lap_running_frames}) -> {os.path.basename(summary_path)}")
                        else:
                            print(f"Discarded short lap (active frames={lap_running_frames} < {min_lap_frames}). No summary created.")

                        # Reset lap state
                        lap_active = False
                        pending_start = False
                        stance_scorer = None
                        posture_counts = None
                        strike_counts = None

            # --------- Drawing overlays ----------
            edges = [
                ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'),
                ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
            ]
            for (a, b) in edges:
                pa = lms_dict.get(a); pb = lms_dict.get(b)
                if pa and pb:
                    xa, ya, _ = pa; xb, yb, _ = pb
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (180, 180, 180), 2)

            nose = lms_dict.get('nose')
            if nose:
                nx, ny, _ = nose
                cv2.circle(frame, (int(nx), int(ny)), 6, (255, 255, 255), -1)
                cv2.circle(frame, (int(nx), int(ny)), 9, (0, 0, 0), 1)

            L_points = {k: lms_dict.get(k) for k in ['left_ankle', 'left_heel', 'left_foot_index']}
            R_points = {k: lms_dict.get(k) for k in ['right_ankle', 'right_heel', 'right_foot_index']}
            if all(L_points.values()):
                ax, ay, _ = L_points['left_ankle']; hx, hy, _ = L_points['left_heel']; tx, ty, _ = L_points['left_foot_index']
                pts = np.array([[int(hx), int(hy)], [int(ax), int(ay)], [int(tx), int(ty)]], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, (int(tx), int(ty)), 3, (0, 255, 0), -1)
            if all(R_points.values()):
                ax, ay, _ = R_points['right_ankle']; hx, hy, _ = R_points['right_heel']; tx, ty, _ = R_points['right_foot_index']
                pts = np.array([[int(hx), int(hy)], [int(ax), int(ay)], [int(tx), int(ty)]], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
                cv2.circle(frame, (int(tx), int(ty)), 3, (0, 165, 255), -1)

            def put_text(img, x, y, label, value=""):
                text = f"{label}: {value}"
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            put_text(frame, 10, 24, "Activity", activity_state)
            put_text(frame, 10, 44, "Posture", features.get('posture_label') or "-")
            put_text(frame, 10, 64, "Lean", f"{features['trunk_lean_deg']:.1f}" if features['trunk_lean_deg'] is not None else "-")
            put_text(frame, 10, 84, "L-Shank", f"{features['L_shank_deg']:.1f}" if features['L_shank_deg'] is not None else "-")
            put_text(frame, 10, 104, "R-Shank", f"{features['R_shank_deg']:.1f}" if features['R_shank_deg'] is not None else "-")
            put_text(frame, 10, 124, "L-Strike", features.get('L_strike') or "-")
            put_text(frame, 10, 144, "R-Strike", features.get('R_strike') or "-")

            if lap_active and activity_state in ("running", "jogging"):
                if (s_post < 0 or s_sh < 0 or s_fs < 0 or s_fp < 0):
                    flagged = True
            if flagged:
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 4)

            out_writer.write(frame)
            if flagged and SAVE_FLAGGED_FRAMES:
                flag_dir = os.path.join(output_dir, "flagged_frames")
                os.makedirs(flag_dir, exist_ok=True)
                cv2.imwrite(os.path.join(flag_dir, f"frame_{frame_idx:05d}.jpg"), frame)

            csv_writer.writerow([
                frame_idx, activity_state, features.get('posture_label') or "",
                f"{features.get('trunk_lean_deg'):.2f}" if features.get('trunk_lean_deg') is not None else "",
                f"{features.get('L_shank_deg'):.2f}" if features.get('L_shank_deg') is not None else "",
                f"{features.get('R_shank_deg'):.2f}" if features.get('R_shank_deg') is not None else "",
                f"{features.get('L_foot_pitch_deg'):.2f}" if features.get('L_foot_pitch_deg') is not None else "",
                f"{features.get('R_foot_pitch_deg'):.2f}" if features.get('R_foot_pitch_deg') is not None else "",
                features.get('L_strike') or "", features.get('R_strike') or ""
            ])

            frame_idx += 1

    cap.release(); out_writer.release(); csv_file.close()

    # If a lap is mid-run when video ends, finalize it conditionally
    try:
        lap_active  # name exists?
        stance_scorer
    except NameError:
        lap_active = False
        stance_scorer = None

    if lap_active and stance_scorer is not None:
        lap_end_frame = frame_idx - 1
        lap_total_frames = lap_end_frame - lap_start_frame_idx + 1
        lap_running_frames = locals().get("lap_running_frames", 0)

        if lap_running_frames >= min_lap_frames:
            saved_lap_count = locals().get("saved_lap_count", 0) + 1
            report = stance_scorer.finalize()
            report["frames"] = lap_total_frames
            report["active_running_frames"] = lap_running_frames
            report["posture_distribution"] = posture_counts or {"Straight":0, "Curved forward":0, "Arched (backward)":0}
            report["footstrike_distribution"] = strike_counts or {"L":{"heel":0,"midfoot":0,"forefoot":0},
                                                                  "R":{"heel":0,"midfoot":0,"forefoot":0}}
            # Recommendations (shared)
            report["recommendations"] = build_recommendations(report["segment_raw"], report.get("session_notes", {}), posture_counts)

            summary_path = os.path.join(output_dir, f"summary_lap_{saved_lap_count}.json")
            with open(summary_path, "w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=2)
            print(f"Saved lap #{saved_lap_count} at end of video (frames={lap_total_frames}, active={lap_running_frames}) -> {os.path.basename(summary_path)}")
        else:
            print(f"Discarded short lap at end of video (active frames={lap_running_frames} < {min_lap_frames}). No summary created.")

    # If no accepted laps at all, create a single “no running” summary
    if saved_lap_count == 0:
        total_frames = frame_idx
        seg_raw = {"posture": 0.0, "shank": 0.0, "foot_strike": 0.0, "foot_pitch": 0.0, "msa": 0.0, "symmetry": 0.0}
        weights = {"posture": 1.0, "shank": 0.8, "foot_strike": 1.0, "foot_pitch": 0.6, "msa": 1.2, "symmetry": 1.0}
        seg_weighted = {k: seg_raw[k] * weights[k] for k in seg_raw}
        total_score = sum(seg_weighted.values())
        notes = {"MSA_L_max": 0.0, "MSA_R_max": 0.0}
        report = {
            "segment_raw": seg_raw,
            "segment_weighted": seg_weighted,
            "counts": {"posture": 0, "shank": 0, "foot_strike": 0, "foot_pitch": 0, "msa": 0, "symmetry": 0},
            "session_notes": notes,
            "total_score": total_score,
            "frames": total_frames,
            "posture_distribution": {"Straight": 0, "Curved forward": 0, "Arched (backward)": 0},
            "footstrike_distribution": {"L": {"heel": 0, "midfoot": 0, "forefoot": 0},
                                         "R": {"heel": 0, "midfoot": 0, "forefoot": 0}},
            "recommendations": ["No running/jogging segments (≥3 s) detected in this video."]
        }
        summary_path = os.path.join(output_dir, "summary_lap_1.json")
        with open(summary_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        print("No accepted laps -> summary_lap_1.json created (informational).")

# ---------- Robust realtime capture opener (Windows-friendly) ----------
def _open_capture_flex(source):
    info = {"backend": None, "device_index": None, "fourcc_set": False}

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str) and not str(source).isdigit():
        c = cv2.VideoCapture(source)
        if c.isOpened():
            info["backend"] = "default-url/path"
            return c, info
        return None, info

    dev_candidates = [source]
    if isinstance(source, int):
        dev_candidates += [i for i in (0, 1, 2, 3) if i != source]

    system = platform.system().lower()
    if "windows" in system:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for dev in dev_candidates:
        for be in backends:
            try:
                c = cv2.VideoCapture(dev, be)
            except Exception:
                c = cv2.VideoCapture(dev)
            if c.isOpened():
                info["backend"] = {cv2.CAP_DSHOW:"CAP_DSHOW", cv2.CAP_MSMF:"CAP_MSMF", cv2.CAP_ANY:"CAP_ANY"}.get(be, str(be))
                info["device_index"] = dev
                try:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    c.set(cv2.CAP_PROP_FOURCC, fourcc)
                    info["fourcc_set"] = True
                except Exception:
                    pass
                try:
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return c, info

    return None, info

# ---------- Realtime processing (with live summary + recommendations) ----------
def process_realtime(source=0, output_dir=None, save_video=False, save_csv=False):
    """
    Realtime mode using webcam/stream/file.
    - source: 0 (default webcam), video path, or URL
    - output_dir: optional folder to save annotated video & CSV and live summary
    - save_video / save_csv: toggle recording
    """
    cap, caminfo = _open_capture_flex(source)
    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Cannot open realtime source: {source}. "
            f"Tips: try --source 1 (or 2), close other apps, or check Windows camera privacy."
        )
    print(f"[realtime] camera opened with backend={caminfo.get('backend')} index={caminfo.get('device_index')} fourcc_set={caminfo.get('fourcc_set')}")

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    FPS = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if FPS < 1.0 or FPS > 240.0:
        FPS = 30.0

    writer = None
    csv_writer = None
    csv_file = None
    csv_rows_since_flush = 0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if save_video and output_dir:
        out_mp4 = os.path.join(output_dir, "annotated_realtime.mp4")
        out_avi = os.path.join(output_dir, "annotated_realtime.avi")
        fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4, fourcc_mp4v, FPS, (W, H))
        if not writer.isOpened():
            fourcc_xvid = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(out_avi, fourcc_xvid, FPS, (W, H))
            if writer.isOpened():
                print(f"[realtime] Recording VIDEO to: {out_avi} (XVID)")
            else:
                print("[realtime] WARNING: VideoWriter could not open MP4 or AVI. Video will NOT be saved.")
                writer = None
        else:
            print(f"[realtime] Recording VIDEO to: {out_mp4} (mp4v)")

    if save_csv and output_dir:
        csv_path = os.path.join(output_dir, "features_realtime.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "t_sec", "activity", "posture_label",
            "trunk_lean_deg", "L_shank_deg", "R_shank_deg",
            "L_foot_pitch_deg", "R_foot_pitch_deg", "L_strike", "R_strike"
        ])
        print(f"[realtime] Recording CSV to: {csv_path}")

    live_summary_path = os.path.join(output_dir, "live_summary.json") if output_dir else None
    last_live_write = 0.0
    saved_lap_count = 0

    def write_live_summary(activity_state, lap_active, lap_frames, posture_counts, strike_counts, scorer_snapshot, force=False):
        nonlocal last_live_write
        now = time.time()
        if live_summary_path is None:
            return
        if not force and (now - last_live_write < 1.0):
            return

        # Build provisional recommendations from partial segment_raw
        recs_so_far = []
        if scorer_snapshot and scorer_snapshot.get("frames_scored", 0) > 0:
            seg_raw_partial = scorer_snapshot.get("segment_raw_partial", {})
            # inject zeros for keys the partial doesn't have
            seg_raw_for_recs = {
                "posture": seg_raw_partial.get("posture", 0.0),
                "shank": seg_raw_partial.get("shank", 0.0),
                "foot_strike": seg_raw_partial.get("foot_strike", 0.0),
                "foot_pitch": seg_raw_partial.get("foot_pitch", 0.0),
                "msa": 0.0, "symmetry": 0.0
            }
            recs_so_far = build_recommendations(seg_raw_for_recs, notes=None, posture_counts=posture_counts)

        doc = {
            "activity": activity_state,
            "laps_completed": saved_lap_count,
            "lap_active": bool(lap_active),
            "active_running_frames": int(lap_frames) if lap_frames is not None else 0,
            "posture_distribution_so_far": posture_counts or {"Straight":0, "Curved forward":0, "Arched (backward)":0},
            "footstrike_distribution_so_far": strike_counts or {"L":{"heel":0,"midfoot":0,"forefoot":0},
                                                                "R":{"heel":0,"midfoot":0,"forefoot":0}},
            "scorer_snapshot": scorer_snapshot or {},
            "recommendations_so_far": recs_so_far
        }
        try:
            with open(live_summary_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            last_live_write = now
        except Exception as e:
            print(f"[realtime] WARNING: cannot write live_summary.json ({e})")

    detector = RunningDetector(FPS, window_sec=4.0)
    prev_state = {}

    lap_active = False
    pending_start = False
    pending_frames = 0
    break_frames = 0
    confirm_run_frames = int(np.ceil(0.5 * FPS))
    max_break_frames = int(np.ceil(3.0 * FPS))
    min_lap_frames   = int(np.ceil(3.0 * FPS))

    stance_scorer = None
    lap_running_frames = 0
    posture_counts = None
    strike_counts = None

    import mediapipe as mp
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                enable_segmentation=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as pose:

        t0 = time.time()
        frame_idx = 0
        fps_ema = None
        last_ts = time.time()

        while True:
            ok = cap.grab()
            if not ok:
                if isinstance(source, str) and os.path.isfile(source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            ok, frame = cap.retrieve()
            if not ok:
                break

            s_post = s_sh = s_fs = s_fp = 0.0
            flagged = False

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            lms_dict = {}
            if results.pose_landmarks:
                for name, idx in POSE_LM.items():
                    lm = results.pose_landmarks.landmark[idx]
                    lms_dict[name] = (lm.x * W, lm.y * H, float(getattr(lm, "visibility", 0.0)))

            features, prev_state = extract_running_features(lms_dict, W, H, prev_state)
            act_info = detector.update(features)
            activity_state = act_info["state"]

            # Lap logic
            if not lap_active:
                if not pending_start and activity_state in ("running", "jogging"):
                    pending_start = True
                    pending_frames = 1
                elif pending_start:
                    if activity_state in ("running", "jogging"):
                        pending_frames += 1
                        if pending_frames >= confirm_run_frames:
                            lap_active = True
                            pending_start = False
                            pending_frames = 0
                            break_frames = 0
                            stance_scorer = StanceScorer(fps=FPS)
                            posture_counts = {"Straight": 0, "Curved forward": 0, "Arched (backward)": 0}
                            strike_counts = {"L": {"heel": 0, "midfoot": 0, "forefoot": 0},
                                             "R": {"heel": 0, "midfoot": 0, "forefoot": 0}}
                            lap_running_frames = 0
                            # immediate live write on lap start (no recs yet)
                            write_live_summary(activity_state, True, 0, posture_counts, strike_counts, {}, force=True)
                    else:
                        pending_start = False
                        pending_frames = 0
            else:
                if activity_state in ("running", "jogging"):
                    break_frames = 0
                    stance_scorer.update_frame(features, activity_state)
                    lap_running_frames += 1

                    s_post = stance_scorer.details[-1]["posture_score"]
                    s_sh   = stance_scorer.details[-1]["shank_score"]
                    s_fs   = stance_scorer.details[-1]["foot_strike_score"]
                    s_fp   = stance_scorer.details[-1]["foot_pitch_score"]
                    flagged = (s_post < 0 or s_sh < 0 or s_fs < 0 or s_fp < 0)

                    pb = stance_scorer.details[-1]["posture_bucket"]
                    if pb and pb != "N/A":
                        if pb in ("Straight", "Curved forward", "Arched (backward)"):
                            posture_counts[pb] += 1
                        elif pb in ("Slight arch", "Arched"):
                            posture_counts["Arched (backward)"] += 1
                        elif pb == "Too forward":
                            posture_counts["Curved forward"] += 1
                    Ls = features.get('L_strike'); Rs = features.get('R_strike')
                    if Ls and Ls in strike_counts["L"]:
                        strike_counts["L"][Ls] += 1
                    if Rs and Rs in strike_counts["R"]:
                        strike_counts["R"][Rs] += 1
                else:
                    break_frames += 1
                    if break_frames > max_break_frames:
                        if lap_running_frames >= min_lap_frames:
                            rpt = stance_scorer.finalize()
                            saved_lap_count += 1
                            # finalize recommendations and write to live summary
                            final_recs = build_recommendations(rpt.get("segment_raw", {}), rpt.get("session_notes", {}), posture_counts)
                            if live_summary_path:
                                try:
                                    with open(live_summary_path, "w", encoding="utf-8") as f:
                                        json.dump({
                                            "activity": activity_state,
                                            "laps_completed": saved_lap_count,
                                            "lap_active": False,
                                            "last_lap_total_score": float(rpt.get("total_score", 0.0)),
                                            "last_lap_session_notes": rpt.get("session_notes", {}),
                                            "last_lap_recommendations": final_recs
                                        }, f, indent=2)
                                    last_live_write = time.time()
                                except Exception:
                                    pass
                            print("[lap] ended (active_frames=", lap_running_frames,
                                  ", total_score=", round(rpt["total_score"], 2), ")")
                        # reset
                        lap_active = False
                        stance_scorer = None
                        posture_counts = None
                        strike_counts = None
                        lap_running_frames = 0

            # --- draw overlays ---
            edges = [
                ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'),
                ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
                ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
            ]
            for (a, b) in edges:
                pa = lms_dict.get(a); pb = lms_dict.get(b)
                if pa and pb:
                    xa, ya, _ = pa; xb, yb, _ = pb
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (180, 180, 180), 2)

            def put_text(img, x, y, label, value=""):
                text = f"{label}: {value}"
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

            now = time.time()
            inst_fps = 1.0 / max(1e-6, (now - last_ts))
            fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)
            last_ts = now

            put_text(frame, 10, 24, "Activity", activity_state)
            put_text(frame, 10, 44, "FPS", f"{fps_ema:.1f}")
            if lap_active:
                put_text(frame, 10, 64, "LapFrames", str(lap_running_frames))

            if lap_active and (s_post < 0 or s_sh < 0 or s_fs < 0 or s_fp < 0):
                flagged = True
            if flagged:
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 3)

            if writer is not None:
                cv2.circle(frame, (W - 20, 20), 6, (0, 0, 255), -1)
                writer.write(frame)

            cv2.imshow("Xstep Realtime (q/ESC to quit, r to reset)", frame)

            if csv_writer is not None:
                csv_writer.writerow([
                    f"{now - t0:.3f}", activity_state, features.get('posture_label') or "",
                    f"{features.get('trunk_lean_deg'):.2f}" if features.get('trunk_lean_deg') is not None else "",
                    f"{features.get('L_shank_deg'):.2f}"    if features.get('L_shank_deg')    is not None else "",
                    f"{features.get('R_shank_deg'):.2f}"    if features.get('R_shank_deg')    is not None else "",
                    f"{features.get('L_foot_pitch_deg'):.2f}" if features.get('L_foot_pitch_deg') is not None else "",
                    f"{features.get('R_foot_pitch_deg'):.2f}" if features.get('R_foot_pitch_deg') is not None else "",
                    features.get('L_strike') or "", features.get('R_strike') or ""
                ])
                csv_rows_since_flush += 1
                if csv_rows_since_flush >= 30:
                    csv_file.flush()
                    csv_rows_since_flush = 0

            # ---- update live summary (~1 Hz) with recommendations_so_far ----
            scorer_snapshot = stance_scorer.snapshot() if (lap_active and stance_scorer is not None) else {}
            write_live_summary(activity_state, lap_active, lap_running_frames if lap_active else 0,
                               posture_counts, strike_counts, scorer_snapshot, force=False)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('r'):
                detector = RunningDetector(FPS, window_sec=4.0)
                prev_state = {}
                print("[realtime] detector reset")

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if csv_file is not None:
        csv_file.flush(); csv_file.close()
    cv2.destroyAllWindows()

# ---------- CLI ----------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Xstep posture analysis")
    p.add_argument("--mode", choices=["batch", "realtime"], default="batch",
                   help="batch: process files in input/; realtime: webcam/file/stream")
    p.add_argument("--source", default="0",
                   help="realtime source: camera index (e.g., 0) or video/RTSP URL or file path")
    p.add_argument("--out", default="output",
                   help="output folder (used in both modes)")
    p.add_argument("--save-video", action="store_true", help="save annotated video (realtime)")
    p.add_argument("--save-csv", action="store_true", help="save features CSV (realtime)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.mode == "realtime":
        try:
            src = int(args.source)
        except ValueError:
            src = args.source
        process_realtime(source=src, output_dir=args.out,
                         save_video=args.save_video, save_csv=args.save_csv)
        return

    # batch mode
    in_dir = "input" if os.path.isdir("input") else "video"
    if not os.path.isdir(in_dir):
        print("No input/ or video/ folder found. Place .mp4 files in 'input'.")
        sys.exit(0)
    videos = [os.path.join(in_dir, f) for f in sorted(os.listdir(in_dir)) if f.lower().endswith(".mp4")]
    if not videos:
        print("No .mp4 files found in", in_dir)
        sys.exit(0)
    for i, vid_path in enumerate(videos, start=1):
        out_dir = os.path.join(args.out, f"output_{i}")
        print(f"\n=== Processing #{i}: {vid_path} -> {out_dir}")
        process_video(vid_path, out_dir)

if __name__ == "__main__":
    main()

