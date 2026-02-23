# pose_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# All pose-estimation logic: config, model loading, helpers, feature extraction,
# kick detection, and CSV persistence.
#
# IMPROVEMENTS:
#  - Dynamic kicking leg detection (auto or explicit left/right)
#  - Selects the BEST kick (highest ankle speed) instead of the first
#  - Exports the annotated frame as a base64 PNG for the frontend
#  - Cleaner config via environment variables
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import math
import base64
import threading
import logging
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH             = os.getenv("MODEL_PATH", "3.tflite")
CONF_THRESH            = float(os.getenv("CONF_THRESH", "0.20"))   # lowered: captures more keypoints in varied lighting
# "auto" → pick the leg with highest ankle speed; "right" / "left" forces a side
KICKING_LEG            = os.getenv("KICKING_LEG", "auto")

# Primary detection gate: ankle must move faster than this (px/s).
# Lowered from 360 to handle slow-mo, phone videos, and partial kicks.
KICK_THRESHOLD_PPS     = float(os.getenv("KICK_THRESHOLD_PPS",     "120"))
# Secondary gate: knee angular velocity.  Now OR-gated with ankle speed (see below).
KNEE_ANG_VEL_THRESHOLD = float(os.getenv("KNEE_ANG_VEL_THRESHOLD", "80"))
KICK_COOLDOWN_SECS     = float(os.getenv("KICK_COOLDOWN_SECS",     "0.4"))
SMOOTH_ALPHA           = float(os.getenv("SMOOTH_ALPHA",           "0.35"))
FPS_ESTIMATE           = float(os.getenv("FPS_ESTIMATE",           "30.0"))
MAX_KICKS_TO_TRACK     = int(os.getenv("MAX_KICKS_TO_TRACK",       "10"))  # wider net

MAX_UPLOAD_BYTES  = 200 * 1024 * 1024
ALLOWED_MIME_PREFIX = "video/"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - KickCoach - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TFLite interpreter  (loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────
interpreter    = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("TFLite interpreter loaded and tensors allocated")

interp_lock = threading.Lock()

# Warm-up
try:
    _dummy = np.zeros((1, 192, 192, 3), dtype=np.float32)
    with interp_lock:
        interpreter.set_tensor(input_details[0]["index"], _dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
    logger.info("Interpreter warm-up complete")
except Exception as _e:
    logger.exception("Interpreter warm-up failed: %s", _e)

# ─────────────────────────────────────────────────────────────────────────────
# SKELETON
# ─────────────────────────────────────────────────────────────────────────────
EDGES = {
    (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c",
    (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m",
    (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
    (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m",
    (12, 14): "c", (14, 16): "c",
}

LSH,  RSH   = 5,  6
LHIP, RHIP  = 11, 12
LKNEE,RKNEE = 13, 14
LANKLE,RANKLE = 15, 16

# Colour map for skeleton drawing
_COLOUR_MAP = {"m": (255, 0, 255), "c": (0, 255, 255), "y": (0, 255, 0)}

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def scaled_keypoints_from_output(keypoints, frame_shape):
    h, w, _ = frame_shape
    shaped = np.squeeze(keypoints)
    pixel_kps = np.zeros_like(shaped, dtype=np.float32)
    for i in range(shaped.shape[0]):
        y_norm, x_norm, sc = shaped[i]
        pixel_kps[i, 0] = float(y_norm * h)
        pixel_kps[i, 1] = float(x_norm * w)
        pixel_kps[i, 2] = float(sc)
    return pixel_kps


def angle_between_points(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    BA, BC = a - b, c - b
    nBA, nBC = np.linalg.norm(BA), np.linalg.norm(BC)
    if nBA < 1e-6 or nBC < 1e-6:
        return np.nan
    cos_ang = np.clip(np.dot(BA, BC) / (nBA * nBC), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def trunk_tilt_signed_degrees(shoulder_l, shoulder_r, hip_l, hip_r):
    try:
        sh_mid_x  = (shoulder_l[0] + shoulder_r[0]) / 2.0
        sh_mid_y  = (shoulder_l[1] + shoulder_r[1]) / 2.0
        hip_mid_x = (hip_l[0]      + hip_r[0])      / 2.0
        hip_mid_y = (hip_l[1]      + hip_r[1])      / 2.0
    except Exception:
        return np.nan
    dx, dy = sh_mid_x - hip_mid_x, sh_mid_y - hip_mid_y
    v_norm = math.hypot(dx, dy)
    if v_norm < 1e-6:
        return np.nan
    return float(math.degrees(math.atan2(dx, -dy)))


def torso_pelvis_twist_2d(sh_l, sh_r, hip_l, hip_r):
    if None in (sh_l, sh_r, hip_l, hip_r):
        return np.nan
    v_sh  = np.array([sh_r[0] - sh_l[0],  sh_r[1] - sh_l[1]])
    v_hip = np.array([hip_r[0] - hip_l[0], hip_r[1] - hip_l[1]])
    n1, n2 = np.linalg.norm(v_sh), np.linalg.norm(v_hip)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cos_ang = np.clip(np.dot(v_sh, v_hip) / (n1 * n2), -1.0, 1.0)
    angle   = np.degrees(np.arccos(cos_ang))
    cross   = np.cross(v_sh, v_hip)
    return float(angle if cross > 0 else -angle)

# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.v = None

    def update(self, x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return self.v
        if self.v is None:
            self.v = float(x)
        else:
            self.v = float(self.alpha * x + (1 - self.alpha) * self.v)
        return self.v

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def nan_to_none(x):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except Exception:
        pass
    return x


def save_final_snapshot_csv(snapshot: dict, fname: str = "final_kick_features.csv"):
    out = {k: ("" if v is None else v) for k, v in snapshot.items() if k != "frame_b64"}
    out["captured_at"] = datetime.now().isoformat()
    fieldnames   = list(out.keys())
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(out)


def _draw_skeleton(frame: np.ndarray, pixel_kps: np.ndarray,
                   highlight_indices: list[int]) -> np.ndarray:
    """Draw skeleton edges and keypoints on a copy of frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    for (p1, p2), colour_key in EDGES.items():
        c1 = pixel_kps[p1, 2]
        c2 = pixel_kps[p2, 2]
        if c1 < CONF_THRESH or c2 < CONF_THRESH:
            continue
        x1, y1 = int(pixel_kps[p1, 1]), int(pixel_kps[p1, 0])
        x2, y2 = int(pixel_kps[p2, 1]), int(pixel_kps[p2, 0])
        cv2.line(out, (x1, y1), (x2, y2), _COLOUR_MAP.get(colour_key, (255, 255, 255)), 2)

    for i, (y, x, conf) in enumerate(pixel_kps):
        if conf < CONF_THRESH:
            continue
        color = (0, 200, 255) if i in highlight_indices else (200, 200, 200)
        radius = 6 if i in highlight_indices else 4
        cv2.circle(out, (int(x), int(y)), radius, color, -1)

    return out


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode a BGR frame as JPEG base64 string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _detect_kicking_leg(pixel_kps: np.ndarray) -> str:
    """
    Heuristic: whichever ankle is lower in the frame (higher y) is
    more likely the planted foot; the other is the kicking foot.
    Returns 'right' or 'left'.
    """
    r_ankle_y = pixel_kps[RANKLE, 0]
    l_ankle_y = pixel_kps[LANKLE, 0]
    r_conf    = pixel_kps[RANKLE, 2]
    l_conf    = pixel_kps[LANKLE, 2]
    # The kicking ankle rises; the planted ankle stays lower.
    # If right ankle is HIGHER (smaller y), right foot is kicking.
    if r_conf < CONF_THRESH and l_conf < CONF_THRESH:
        return "right"  # fallback
    if r_conf < CONF_THRESH:
        return "left"
    if l_conf < CONF_THRESH:
        return "right"
    return "right" if r_ankle_y < l_ankle_y else "left"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_peak_frame(video_path: str, legs_to_check: list[str]) -> dict | None:
    """
    Scan the video and return a snapshot at the frame where ankle speed
    peaks — used when normal detection thresholds are not met.
    This ensures phones, slow-mo, or short clips always get analysed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ESTIMATE

    knee_ema  = {leg: EMA(SMOOTH_ALPHA) for leg in legs_to_check}
    trunk_ema = EMA(SMOOTH_ALPHA)
    hip_ema   = EMA(SMOOTH_ALPHA)

    prev_ankles = {leg: None for leg in legs_to_check}
    prev_knees  = {leg: None for leg in legs_to_check}

    peak_speed    = -1.0
    peak_snapshot = None
    peak_frame    = None
    peak_kps      = None
    peak_leg      = legs_to_check[0]
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
            with interp_lock:
                interpreter.set_tensor(input_details[0]["index"], np.array(tf.cast(img, dtype=tf.float32)))
                interpreter.invoke()
                kps_norm = interpreter.get_tensor(output_details[0]["index"])
        except Exception:
            frame_idx += 1
            continue

        pixel_kps = scaled_keypoints_from_output(kps_norm, frame.shape)

        def to_xy_conf(idx):
            y, x, c = pixel_kps[idx]
            return (float(x), float(y)), float(c)

        sh_l_pt, sh_l_conf = to_xy_conf(LSH)
        sh_r_pt, sh_r_conf = to_xy_conf(RSH)
        hp_l_pt, hp_l_conf = to_xy_conf(LHIP)
        hp_r_pt, hp_r_conf = to_xy_conf(RHIP)

        trunk_val = np.nan
        if min(sh_l_conf, sh_r_conf, hp_l_conf, hp_r_conf) >= CONF_THRESH:
            trunk_val = trunk_tilt_signed_degrees(sh_l_pt, sh_r_pt, hp_l_pt, hp_r_pt)
        hip_rot = np.nan
        if min(sh_l_conf, sh_r_conf, hp_l_conf, hp_r_conf) >= CONF_THRESH:
            hip_rot = torso_pelvis_twist_2d(sh_l_pt, sh_r_pt, hp_l_pt, hp_r_pt)

        trunk_ema.update(trunk_val)
        hip_ema.update(hip_rot)

        for leg in legs_to_check:
            if leg == "right":
                hip_idx, knee_idx, ankle_idx = RHIP, RKNEE, RANKLE
            else:
                hip_idx, knee_idx, ankle_idx = LHIP, LKNEE, LANKLE

            hip_pt,   hip_conf   = to_xy_conf(hip_idx)
            knee_pt,  knee_conf  = to_xy_conf(knee_idx)
            ankle_pt, ankle_conf = to_xy_conf(ankle_idx)

            knee_angle = np.nan
            if min(hip_conf, knee_conf, ankle_conf) >= CONF_THRESH:
                knee_angle = angle_between_points(hip_pt, knee_pt, ankle_pt)

            ankle_speed = np.nan
            if prev_ankles[leg] is not None and ankle_conf >= CONF_THRESH:
                dx = ankle_pt[0] - prev_ankles[leg][0]
                dy = ankle_pt[1] - prev_ankles[leg][1]
                ankle_speed = math.hypot(dx, dy) * fps

            knee_kav = np.nan
            if prev_knees[leg] is not None and not np.isnan(knee_angle):
                knee_kav = (knee_angle - prev_knees[leg]) * fps

            prev_ankles[leg] = ankle_pt
            if not np.isnan(knee_angle):
                prev_knees[leg] = knee_angle

            knee_ema[leg].update(knee_angle)

            if isinstance(ankle_speed, float) and not np.isnan(ankle_speed) and ankle_speed > peak_speed:
                peak_speed    = ankle_speed
                peak_leg      = leg
                peak_frame    = frame.copy()
                peak_kps      = pixel_kps.copy()
                peak_snapshot = {
                    "frame_number":     int(frame_idx),
                    "detected_leg":     leg,
                    "knee_angle":       nan_to_none(knee_ema[leg].v),
                    "trunk":            nan_to_none(trunk_ema.v),
                    "hip_rotation":     nan_to_none(hip_ema.v),
                    "ankle_speed_pps":  nan_to_none(ankle_speed),
                    "knee_ang_vel_dps": nan_to_none(knee_kav),
                }

        frame_idx += 1

    cap.release()

    if peak_snapshot is None or peak_frame is None:
        return None

    highlight = [RHIP, RKNEE, RANKLE] if peak_leg == "right" else [LHIP, LKNEE, LANKLE]
    annotated = _draw_skeleton(peak_frame, peak_kps, highlight)
    peak_snapshot["frame_b64"] = _frame_to_b64(annotated)

    logger.info(
        "Fallback: peak ankle speed %.0f px/s at frame %d (leg=%s)",
        peak_speed, peak_snapshot["frame_number"], peak_leg,
    )
    return peak_snapshot


def analyze_video_file(
    video_path: str,
    kicking_leg: str = "auto",
    max_kicks: int   = MAX_KICKS_TO_TRACK,
) -> dict | None:
    """
    Process a video file and return the BEST detected kick snapshot, or None.

    Parameters
    ----------
    video_path   : str  – absolute path to the video file.
    kicking_leg  : str  – "auto" | "right" | "left"
    max_kicks    : int  – how many kick candidates to collect before picking best.

    Returns
    -------
    dict | None  – kick feature dictionary or None.
        Extra keys vs old version:
          detected_leg  – which leg was analysed ("right" / "left")
          frame_b64     – JPEG base64 of the annotated kick frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video file: %s", video_path)
        raise ValueError(f"Could not open video: {video_path}")

    fps                  = cap.get(cv2.CAP_PROP_FPS) or FPS_ESTIMATE
    kick_cooldown_frames = max(1, int(round(KICK_COOLDOWN_SECS * fps)))
    kick_cooldown        = 0

    # EMA per possible leg
    state = {
        "right": {
            "knee_ema": EMA(SMOOTH_ALPHA), "trunk_ema": EMA(SMOOTH_ALPHA),
            "hip_ema": EMA(SMOOTH_ALPHA),
            "prev_ankle": None, "prev_knee_angle": None,
        },
        "left": {
            "knee_ema": EMA(SMOOTH_ALPHA), "trunk_ema": EMA(SMOOTH_ALPHA),
            "hip_ema": EMA(SMOOTH_ALPHA),
            "prev_ankle": None, "prev_knee_angle": None,
        },
    }

    kick_candidates: list[dict] = []   # collect up to max_kicks
    frame_store: dict[int, np.ndarray] = {}   # frame_idx → raw frame for overlay
    frame_idx = 0

    while len(kick_candidates) < max_kicks:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            img         = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
            input_image = tf.cast(img, dtype=tf.float32)
            with interp_lock:
                interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
                interpreter.invoke()
                kps_norm = interpreter.get_tensor(output_details[0]["index"])
        except Exception as inf_e:
            logger.exception("TFLite inference error at frame %d: %s", frame_idx, inf_e)
            frame_idx += 1
            continue

        pixel_kps = scaled_keypoints_from_output(kps_norm, frame.shape)

        # Determine which legs to analyse this frame
        if kicking_leg.lower() == "auto":
            legs_to_check = ["right", "left"]
        else:
            legs_to_check = [kicking_leg.lower()]

        def to_xy_conf(idx):
            y, x, c = pixel_kps[idx]
            return (float(x), float(y)), float(c)

        sh_l_pt,  sh_l_conf  = to_xy_conf(LSH)
        sh_r_pt,  sh_r_conf  = to_xy_conf(RSH)
        hip_l_pt, hip_l_conf = to_xy_conf(LHIP)
        hip_r_pt, hip_r_conf = to_xy_conf(RHIP)

        # Trunk & hip rotation (same for both legs)
        trunk_val = np.nan
        if min(sh_l_conf, sh_r_conf, hip_l_conf, hip_r_conf) >= CONF_THRESH:
            trunk_val    = trunk_tilt_signed_degrees(sh_l_pt, sh_r_pt, hip_l_pt, hip_r_pt)
        hip_rotation = np.nan
        if min(sh_l_conf, sh_r_conf, hip_l_conf, hip_r_conf) >= CONF_THRESH:
            hip_rotation = torso_pelvis_twist_2d(sh_l_pt, sh_r_pt, hip_l_pt, hip_r_pt)

        for leg in legs_to_check:
            s = state[leg]
            if leg == "right":
                hip_idx, knee_idx, ankle_idx = RHIP, RKNEE, RANKLE
            else:
                hip_idx, knee_idx, ankle_idx = LHIP, LKNEE, LANKLE

            hip_pt,   hip_conf   = to_xy_conf(hip_idx)
            knee_pt,  knee_conf  = to_xy_conf(knee_idx)
            ankle_pt, ankle_conf = to_xy_conf(ankle_idx)

            # Knee angle
            knee_angle = np.nan
            if min(hip_conf, knee_conf, ankle_conf) >= CONF_THRESH:
                knee_angle = angle_between_points(hip_pt, knee_pt, ankle_pt)

            # Ankle speed
            ankle_speed_pps = np.nan
            if s["prev_ankle"] is not None and ankle_conf >= CONF_THRESH:
                dx = ankle_pt[0] - s["prev_ankle"][0]
                dy = ankle_pt[1] - s["prev_ankle"][1]
                ankle_speed_pps = math.hypot(dx, dy) * fps
            s["prev_ankle"] = ankle_pt

            # Knee angular velocity
            knee_ang_vel_dps = np.nan
            if s["prev_knee_angle"] is not None and not np.isnan(knee_angle):
                knee_ang_vel_dps = (knee_angle - s["prev_knee_angle"]) * fps
            if not np.isnan(knee_angle):
                s["prev_knee_angle"] = knee_angle

            # Smooth
            sm_knee  = s["knee_ema"].update(knee_angle)
            sm_trunk = s["trunk_ema"].update(trunk_val)
            sm_hip   = s["hip_ema"].update(hip_rotation)

        # ── Kick detection (evaluate best leg or chosen leg) ──────────────
        if kick_cooldown > 0:
            kick_cooldown -= 1
            frame_idx += 1
            continue

        best_leg      = None
        best_speed    = -1.0
        best_kav      = 0.0
        best_score    = -1.0   # composite detection score

        for leg in legs_to_check:
            s = state[leg]
            if leg == "right":
                hip_idx, knee_idx, ankle_idx = RHIP, RKNEE, RANKLE
            else:
                hip_idx, knee_idx, ankle_idx = LHIP, LKNEE, LANKLE

            hip_pt,   hip_conf   = to_xy_conf(hip_idx)
            knee_pt,  knee_conf  = to_xy_conf(knee_idx)
            ankle_pt, ankle_conf = to_xy_conf(ankle_idx)

            knee_angle = np.nan
            if min(hip_conf, knee_conf, ankle_conf) >= CONF_THRESH:
                knee_angle = angle_between_points(hip_pt, knee_pt, ankle_pt)

            ankle_speed_pps = np.nan
            if s["prev_ankle"] is not None and ankle_conf >= CONF_THRESH:
                dx = ankle_pt[0] - s["prev_ankle"][0]
                dy = ankle_pt[1] - s["prev_ankle"][1]
                ankle_speed_pps = math.hypot(dx, dy) * fps

            knee_ang_vel_dps = np.nan
            if s["prev_knee_angle"] is not None and not np.isnan(knee_angle):
                knee_ang_vel_dps = (knee_angle - s["prev_knee_angle"]) * fps

            has_speed = (
                isinstance(ankle_speed_pps, float)
                and not np.isnan(ankle_speed_pps)
                and ankle_speed_pps > KICK_THRESHOLD_PPS
            )
            has_kav = (
                isinstance(knee_ang_vel_dps, float)
                and not np.isnan(knee_ang_vel_dps)
                and abs(knee_ang_vel_dps) > KNEE_ANG_VEL_THRESHOLD
            )

            # OR-gated: either signal alone triggers detection.
            # Composite score picks the best candidate across legs and frames.
            if not (has_speed or has_kav):
                continue

            speed_score = (ankle_speed_pps / KICK_THRESHOLD_PPS) if has_speed else 0.0
            kav_score   = (abs(knee_ang_vel_dps) / KNEE_ANG_VEL_THRESHOLD) if has_kav else 0.0
            composite   = speed_score + kav_score

            if composite > best_score:
                best_score = composite
                best_speed = ankle_speed_pps if has_speed else 0.0
                best_kav   = knee_ang_vel_dps if has_kav else 0.0
                best_leg   = leg

        if best_leg is not None:
            s = state[best_leg]
            # Highlight keypoints for the detected kicking leg
            if best_leg == "right":
                highlight = [RHIP, RKNEE, RANKLE]
            else:
                highlight = [LHIP, LKNEE, LANKLE]

            annotated_frame = _draw_skeleton(frame, pixel_kps, highlight)
            frame_b64       = _frame_to_b64(annotated_frame)

            snapshot = {
                "frame_number":      int(frame_idx),
                "detected_leg":      best_leg,
                "knee_angle":        nan_to_none(s["knee_ema"].v),
                "trunk":             nan_to_none(s["trunk_ema"].v),
                "hip_rotation":      nan_to_none(s["hip_ema"].v),
                "ankle_speed_pps":   nan_to_none(best_speed),
                "knee_ang_vel_dps":  nan_to_none(best_kav),
                "frame_b64":         frame_b64,
            }
            kick_candidates.append(snapshot)
            kick_cooldown = kick_cooldown_frames
            logger.info(
                "Kick candidate #%d detected at frame %d (leg=%s, speed=%.0f px/s)",
                len(kick_candidates), frame_idx, best_leg, best_speed,
            )

        frame_idx += 1

    cap.release()

    if not kick_candidates:
        # ── Last-resort fallback ──────────────────────────────────────────
        # If strict detection never fired, return the frame where the ankle
        # moved fastest (covers slow-mo, phone videos, off-angle shots).
        logger.warning(
            "No kick candidate met detection thresholds. "
            "Falling back to peak-ankle-speed frame."
        )
        best = _fallback_peak_frame(video_path, legs_to_check if kicking_leg.lower() == "auto" else [kicking_leg.lower()])
        if best is None:
            return None
        return best

    # Pick the best kick: highest composite score (most power signal)
    best = max(kick_candidates, key=lambda k: k.get("ankle_speed_pps") or 0.0)
    logger.info(
        "Best kick selected: frame %d, leg=%s, speed=%.0f px/s (from %d candidates)",
        best["frame_number"], best["detected_leg"],
        best.get("ankle_speed_pps") or 0.0, len(kick_candidates),
    )

    try:
        save_final_snapshot_csv(best)
    except Exception:
        logger.exception("Failed to save final snapshot CSV")

    return best