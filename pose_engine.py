import cv2
import numpy as np
import tensorflow as tf
import math

# ==============================
# CONFIG (edit these)
# ==============================
VIDEO_PATH = "Gk.mp4"  # <<< your video file
MODEL_PATH = "3.tflite"                   # <<< your MoveNet tflite file
CONF_THRESH = 0.35                        # keypoint confidence threshold
KICKING_LEG = "right"                     # "right" or "left"
KICK_THRESHOLD_PPS = 360                  # kick detection threshold in pixels/sec (tune)
KNEE_ANG_VEL_THRESHOLD = 250              # deg/sec threshold for knee angular velocity (tune)
KICK_COOLDOWN_SECS = 0.5                  # seconds to wait after detecting a kick
SMOOTH_ALPHA = 0.35                       # EMA smoothing for angles (0..1) higher = less smoothing

# ==============================
# LOAD MODEL
# ==============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==============================
# SKELETON EDGES + COLORS
# ==============================
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}
COLOR_MAP = {
    'm': (255, 0, 255),   # magenta (BGR)
    'c': (255, 255, 0),   # cyan
    'y': (0, 255, 255)    # yellow
}

# ==============================
# HELPERS
# ==============================
def scaled_keypoints_from_output(keypoints, frame_shape):
    """
    Convert model output (normalized y,x,score) -> pixel coords [y, x, score] (shape (17,3))
    """
    h, w, _ = frame_shape
    shaped = np.squeeze(keypoints)  # (17,3)
    pixel_kps = np.zeros_like(shaped, dtype=np.float32)
    for i in range(shaped.shape[0]):
        y_norm, x_norm, sc = shaped[i]
        pixel_kps[i, 0] = float(y_norm * h)
        pixel_kps[i, 1] = float(x_norm * w)
        pixel_kps[i, 2] = float(sc)
    return pixel_kps


def draw_skeleton_and_keypoints(frame, pixel_kps, conf_thresh=CONF_THRESH):
    """
    Draw all keypoints (small green dots) and skeleton edges with color mapping.
    pixel_kps: (17,3) array with [y, x, score]
    """
    # draw keypoints
    for i in range(17):
        y, x, sc = pixel_kps[i]
        if sc >= conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 0), -1)

    # draw edges
    for (p1, p2), col_key in EDGES.items():
        y1, x1, c1 = pixel_kps[p1]
        y2, x2, c2 = pixel_kps[p2]
        if (c1 >= conf_thresh) and (c2 >= conf_thresh):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            color = COLOR_MAP.get(col_key, (255, 255, 255))
            cv2.line(frame, pt1, pt2, color, 2)


def angle_between_points(a, b, c):
    """
    a,b,c are (x,y). Returns angle at b in degrees [0..180] or np.nan for degenerate.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    BA = a - b
    BC = c - b
    nBA = np.linalg.norm(BA)
    nBC = np.linalg.norm(BC)
    if nBA < 1e-6 or nBC < 1e-6:
        return np.nan
    cos_ang = np.dot(BA, BC) / (nBA * nBC)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def trunk_tilt_signed_degrees(shoulder_l, shoulder_r, hip_l, hip_r):
    """
    Improved trunk tilt: signed angle (degrees).
    We compute vector v = shoulder_mid - hip_mid, then compute signed angle
    between v and vertical up using atan2(det, dot) formula:
      angle = atan2(vx, -vy) in radians -> degrees
    Returns signed degrees (positive = tilt toward +x direction).
    Also returns absolute magnitude.
    """
    try:
        sh_mid_x = (shoulder_l[0] + shoulder_r[0]) / 2.0
        sh_mid_y = (shoulder_l[1] + shoulder_r[1]) / 2.0
        hip_mid_x = (hip_l[0] + hip_r[0]) / 2.0
        hip_mid_y = (hip_l[1] + hip_r[1]) / 2.0
    except Exception:
        return (np.nan, np.nan)

    dx = sh_mid_x - hip_mid_x
    dy = sh_mid_y - hip_mid_y
    v_norm = math.hypot(dx, dy)
    if v_norm < 1e-6:
        return (np.nan, np.nan)

    # signed angle: atan2(det, dot) where det(up, v) = v_x and dot(up, v) = -v_y
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    return (float(angle_deg), float(abs(angle_deg)))

# ==============================
# MAIN LIVE LOOP
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
prev_ankle = None
prev_knee_angle = None
frame_count = 0

# compute cooldown frames from seconds (use at runtime since fps known)
KICK_COOLDOWN_FRAMES = max(5, int(round(KICK_COOLDOWN_SECS * fps)))
kick_cooldown = 0

# Joint indices mapping
LHIP, RHIP = 11, 12
LKNEE, RKNEE = 13, 14
LANKLE, RANKLE = 15, 16
LSH, RSH = 5, 6

# UI params (smaller, unobtrusive)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICK = 1
# TEXT_COLOR = (200, 255, 200)   # pale green-ish (features)
# SECOND_COLOR = (255, 230, 140) # for trunk
# THIRD_COLOR = (220, 180, 200)  # for ankle spd

# TEXT_COLOR   = (30, 150, 30)
# SECOND_COLOR = (180, 140, 40)
# THIRD_COLOR  = (140, 70, 120)

TEXT_COLOR = (10, 70, 10)      # deep forest green
SECOND_COLOR = (90, 60, 10)      # dark mustard / brownish gold
THIRD_COLOR = (70, 30, 60)      # deep plum / dark rose

DEG_WORD = " degrees"          # use word 'degrees' to avoid weird glyphs

# smoothing / history
smoothed_knee = None
smoothed_trunk = None
knee_history = []
trunk_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # MoveNet inference
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    kps_norm = interpreter.get_tensor(output_details[0]['index'])  # (1,1,17,3)

    # normalized -> pixel coords
    pixel_kps = scaled_keypoints_from_output(kps_norm, frame.shape)  # (17,3) y,x,score

    # draw full skeleton/keypoints
    draw_skeleton_and_keypoints(frame, pixel_kps, conf_thresh=CONF_THRESH)

    # pick kicking leg indices
    if KICKING_LEG.lower().startswith("r"):
        hip_idx, knee_idx, ankle_idx = RHIP, RKNEE, RANKLE
    else:
        hip_idx, knee_idx, ankle_idx = LHIP, LKNEE, LANKLE

    # helper to return (x,y), conf
    def to_xy_conf(idx):
        y, x, c = pixel_kps[idx]
        return (float(x), float(y)), float(c)

    hip_pt, hip_conf = to_xy_conf(hip_idx)
    knee_pt, knee_conf = to_xy_conf(knee_idx)
    ankle_pt, ankle_conf = to_xy_conf(ankle_idx)

    # knee angle
    knee_angle = np.nan
    if min(hip_conf, knee_conf, ankle_conf) >= CONF_THRESH:
        knee_angle = angle_between_points(hip_pt, knee_pt, ankle_pt)
        # draw kicking-leg thicker on top for visibility
        cv2.line(frame, (int(hip_pt[0]), int(hip_pt[1])), (int(knee_pt[0]), int(knee_pt[1])), (0, 180, 0), 3)
        cv2.line(frame, (int(knee_pt[0]), int(knee_pt[1])), (int(ankle_pt[0]), int(ankle_pt[1])), (0, 180, 0), 3)

    # trunk angle (signed + magnitude)
    sh_l_pt, sh_l_conf = to_xy_conf(LSH)
    sh_r_pt, sh_r_conf = to_xy_conf(RSH)
    hip_l_pt, hip_l_conf = to_xy_conf(LHIP)
    hip_r_pt, hip_r_conf = to_xy_conf(RHIP)

    trunk_signed = np.nan
    trunk_mag = np.nan
    if min(sh_l_conf, sh_r_conf, hip_l_conf, hip_r_conf) >= CONF_THRESH:
        trunk_signed, trunk_mag = trunk_tilt_signed_degrees(sh_l_pt, sh_r_pt, hip_l_pt, hip_r_pt)

    # ankle speed (px/frame -> px/sec)
    ankle_speed_pxpf = 0.0
    ankle_speed_pps = 0.0
    if prev_ankle is not None and ankle_conf >= CONF_THRESH:
        dx = ankle_pt[0] - prev_ankle[0]
        dy = ankle_pt[1] - prev_ankle[1]
        ankle_speed_pxpf = math.sqrt(dx * dx + dy * dy)
        ankle_speed_pps = ankle_speed_pxpf * fps
    prev_ankle = ankle_pt

    # knee angular velocity deg/sec
    knee_ang_vel_dps = 0.0
    if prev_knee_angle is not None and not np.isnan(knee_angle):
        knee_ang_vel_dps = (knee_angle - prev_knee_angle) * fps
    prev_knee_angle = knee_angle if not np.isnan(knee_angle) else prev_knee_angle

    # ----------------------
    # SMOOTH angles (EMA) and store valid frames for summary
    # ----------------------
    if not np.isnan(knee_angle):
        if smoothed_knee is None:
            smoothed_knee = knee_angle
        else:
            smoothed_knee = SMOOTH_ALPHA * knee_angle + (1 - SMOOTH_ALPHA) * smoothed_knee
        knee_history.append(smoothed_knee)

    if not np.isnan(trunk_mag):
        if smoothed_trunk is None:
            smoothed_trunk = trunk_signed
        else:
            smoothed_trunk = SMOOTH_ALPHA * trunk_signed + (1 - SMOOTH_ALPHA) * smoothed_trunk
        trunk_history.append(smoothed_trunk)

    # ----------------------
    # DRAW FEATURES UI (only features) - moved down and human-friendly labels
    # ----------------------
    x0, y0 = 12, 59
    line_h = 20

    knee_label = "Knee Angle: --" if smoothed_knee is None else f"Knee Angle: {int(round(smoothed_knee))}{DEG_WORD}"
    cv2.putText(frame, knee_label, (x0, y0), FONT, FONT_SCALE, TEXT_COLOR, THICK, cv2.LINE_AA)

    trunk_label = "Trunk Lean: --" if smoothed_trunk is None else f"Trunk Lean: {int(round(smoothed_trunk))}{DEG_WORD}"
    cv2.putText(frame, trunk_label, (x0, y0 + line_h), FONT, FONT_SCALE, SECOND_COLOR, THICK, cv2.LINE_AA)

    cv2.putText(frame, f"Ankle spd: {int(round(ankle_speed_pps))} px/s", (x0, y0 + 2 * line_h), FONT, 0.50, THIRD_COLOR, THICK, cv2.LINE_AA)

    kav_label = "Knee vel: --" if prev_knee_angle is None else f"Knee vel: {int(round(knee_ang_vel_dps))} deg/s"
    cv2.putText(frame, kav_label, (x0, y0 + 3 * line_h), FONT, 0.50, (200, 200, 255), THICK, cv2.LINE_AA)

    # ----------------------
    # IMPROVED KICK DETECTION (unchanged logic but using smoothed angles)
    # ----------------------
    is_fast = ankle_speed_pps > KICK_THRESHOLD_PPS
    is_explosive = abs(knee_ang_vel_dps) > KNEE_ANG_VEL_THRESHOLD

    if kick_cooldown > 0:
        kick_cooldown -= 1

    if (is_fast and is_explosive and kick_cooldown == 0):
        kick_text = "KICK DETECTED!"
        tw, th = cv2.getTextSize(kick_text, FONT, 0.95, 3)[0]
        pos_x = max(10, (w - tw) // 2)
        pos_y = h - 48
        cv2.putText(frame, kick_text, (pos_x, pos_y), FONT, 0.95, (0, 0, 255), 3, cv2.LINE_AA)
        kick_cooldown = KICK_COOLDOWN_FRAMES

    # small near-indicator (optional)
    near_fast = ankle_speed_pps > 0.8 * KICK_THRESHOLD_PPS
    near_ang = abs(knee_ang_vel_dps) > 0.8 * KNEE_ANG_VEL_THRESHOLD
    if near_fast or near_ang:
        cv2.circle(frame, (w - 30, 30), 8, (0, 255, 255), -1)

    # show frame
    cv2.imshow("Football Biomechanics — features only", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# ==============================
# FINAL BEST-FIT SUMMARY (print to console)
# ==============================
import statistics


def summarize(name, arr):
    if not arr:
        print(f"{name}: no valid frames")
        return
    med = statistics.median(arr)
    mean = statistics.mean(arr)
    std = statistics.pstdev(arr)
    print(f"{name}  —  median: {med:.1f} deg, mean: {mean:.1f} deg, std: {std:.1f} deg")


print("\n=== Final biomechanics summary (best-fit) ===")
summarize("Knee Angle (smoothed)", knee_history)
summarize("Trunk Lean (smoothed, signed)", trunk_history)
print("Note: trunk 'signed' shows tilt direction: positive -> tilt toward +x image direction.")
