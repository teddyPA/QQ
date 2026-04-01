"""
FLIR Mosquito Motion Detection — Tracking + Flight Path Visualization
=====================================================================
Preview shows MOTION video (not raw) with:
  - Green bounding boxes around detected bugs
  - Colored trail showing each bug's flight path
  - Trajectory map window showing all paths over time

Outputs per recording session:
  camera_left_TIMESTAMP_raw.mp4      — full raw IR video
  camera_left_TIMESTAMP_motion.mp4   — motion video with trails
  camera_left_TIMESTAMP_tracks.csv   — per-frame bug positions
  camera_left_TIMESTAMP_trajmap.png  — final trajectory map image

Controls:
  Main menu : R = Record   S = Save settings   Q = Quit
  Recording : S = Stop     E = Exposure   G = Gain
              T = Threshold   L = Learning rate

Requirements:
    pip install opencv-python "numpy<2" scipy
    PySpin wheel from /Applications/Spinnaker/pyspin/
"""

import os, json, threading, time, sys, select, csv
from collections import defaultdict, deque
from datetime import datetime

# ── NumPy / dependency guards ─────────────────────────────────────────────────
try:
    import numpy as np
    if int(np.__version__.split(".")[0]) >= 2:
        print("ERROR: NumPy 2.x incompatible with PySpin.")
        print("Fix:  pip install 'numpy<2' --force-reinstall")
        sys.exit(1)
except ImportError:
    print("ERROR: numpy not found."); sys.exit(1)

try:
    import PySpin
except ImportError:
    print("ERROR: PySpin not found."); sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not found.  pip install opencv-python"); sys.exit(1)

# scipy for Hungarian algorithm (optimal blob-to-track matching)
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "save_folder":          os.path.expanduser("~/Desktop/camera_recordings"),
    "camera_0_name":        "camera_left",
    "camera_1_name":        "camera_right",
    "frame_rate":           30.0,
    "exposure_time":        8000.0,
    "gain_db":              0.0,
    "image_width":          1440,
    "image_height":         1080,
    "duration_seconds":     0,
    "preview_scale":        0.5,
    "show_preview":         True,
    # Motion detection
    "bg_learning_rate":     0.005,
    "motion_threshold":     20,
    "min_bug_area":         30,
    "blob_dilation":        3,
    # Tracking
    "trail_length":         60,    # frames of trail to draw (2s at 30fps)
    "max_match_distance":   80,    # max pixels to match a blob to an existing track
    "track_timeout":        10,    # frames before a lost track is retired
}
SETTINGS_FILE = "camera_settings.json"

# Trail colours — up to 12 bugs tracked simultaneously, each gets its own colour
TRAIL_COLORS = [
    (0,   255, 0),    # green
    (0,   200, 255),  # cyan
    (255, 100, 0),    # orange
    (180, 0,   255),  # purple
    (255, 255, 0),    # yellow
    (0,   80,  255),  # blue
    (255, 0,   180),  # pink
    (0,   255, 150),  # mint
    (255, 50,  50),   # red
    (150, 255, 50),   # lime
    (50,  150, 255),  # sky
    (255, 200, 50),   # gold
]

# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS  LOAD / SAVE
# ─────────────────────────────────────────────────────────────────────────────
def settings_path(cfg):
    return os.path.join(cfg["save_folder"], SETTINGS_FILE)

def load_settings():
    cfg = dict(DEFAULTS)
    path = os.path.join(DEFAULTS["save_folder"], SETTINGS_FILE)
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
            print(f"  Settings loaded from {path}")
        except Exception as e:
            print(f"  Could not read settings ({e}) — using defaults")
    return cfg

def save_settings(cfg):
    os.makedirs(cfg["save_folder"], exist_ok=True)
    try:
        with open(settings_path(cfg), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Settings saved → {settings_path(cfg)}")
    except Exception as e:
        print(f"  Could not save settings: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  BUG TRACKER  — assigns persistent IDs across frames
# ─────────────────────────────────────────────────────────────────────────────
class BugTracker:
    def __init__(self, cfg):
        self.cfg         = cfg
        self.next_id     = 0
        self.tracks      = {}   # id → {centroid, trail, age, missing}
        self.color_map   = {}   # id → BGR colour
        self.color_pool  = list(range(len(TRAIL_COLORS)))

    def _assign_color(self, track_id):
        if self.color_pool:
            idx = self.color_pool.pop(0)
        else:
            idx = track_id % len(TRAIL_COLORS)
        self.color_map[track_id] = TRAIL_COLORS[idx]
        return idx

    def _release_color(self, track_id):
        if track_id in self.color_map:
            idx = TRAIL_COLORS.index(self.color_map[track_id])
            if idx not in self.color_pool:
                self.color_pool.insert(0, idx)
            del self.color_map[track_id]

    def update(self, detections):
        """
        detections: list of (cx, cy, area, x, y, w, h)
        Returns list of (track_id, cx, cy, area, x, y, w, h, color)
        """
        max_dist    = self.cfg["max_match_distance"]
        trail_len   = self.cfg["trail_length"]
        timeout     = self.cfg["track_timeout"]
        active_ids  = list(self.tracks.keys())

        # ── Match detections to existing tracks ───────────────────────────
        if active_ids and detections:
            cost = np.zeros((len(active_ids), len(detections)))
            for i, tid in enumerate(active_ids):
                tx, ty = self.tracks[tid]["centroid"]
                for j, (cx, cy, *_) in enumerate(detections):
                    cost[i, j] = np.hypot(tx - cx, ty - cy)

            if HAS_SCIPY:
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                # greedy fallback
                row_ind, col_ind = [], []
                used_cols = set()
                for i in range(len(active_ids)):
                    best_j   = min((j for j in range(len(detections))
                                    if j not in used_cols),
                                   key=lambda j: cost[i, j],
                                   default=None)
                    if best_j is not None:
                        row_ind.append(i); col_ind.append(best_j)
                        used_cols.add(best_j)

            matched_track_idx  = set()
            matched_detect_idx = set()
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= max_dist:
                    tid = active_ids[i]
                    cx, cy, *_ = detections[j]
                    trail = self.tracks[tid]["trail"]
                    trail.append((int(cx), int(cy)))
                    if len(trail) > trail_len:
                        trail.popleft()
                    self.tracks[tid]["centroid"] = (cx, cy)
                    self.tracks[tid]["missing"]  = 0
                    matched_track_idx.add(i)
                    matched_detect_idx.add(j)

            # increment missing counter for unmatched tracks
            for i, tid in enumerate(active_ids):
                if i not in matched_track_idx:
                    self.tracks[tid]["missing"] += 1

            # new tracks for unmatched detections
            for j, det in enumerate(detections):
                if j not in matched_detect_idx:
                    self._new_track(det)
        else:
            # no existing tracks — create new for each detection
            for det in detections:
                self._new_track(det)
            # age out all existing
            for tid in active_ids:
                self.tracks[tid]["missing"] += 1

        # ── Remove timed-out tracks ────────────────────────────────────────
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missing"] > timeout:
                self._release_color(tid)
                del self.tracks[tid]

        # ── Build result list ──────────────────────────────────────────────
        results = []
        for tid, tr in self.tracks.items():
            if tr["missing"] == 0:
                cx, cy = tr["centroid"]
                # find matching detection
                for det in detections:
                    dcx, dcy, area, x, y, w, h = det
                    if abs(dcx - cx) < 2 and abs(dcy - cy) < 2:
                        results.append((tid, cx, cy, area, x, y, w, h,
                                        self.color_map.get(tid, (0, 255, 0))))
                        break
        return results

    def _new_track(self, det):
        cx, cy, area, x, y, w, h = det
        tid   = self.next_id
        self.next_id += 1
        trail = deque([(int(cx), int(cy))], maxlen=self.cfg["trail_length"])
        self.tracks[tid] = {
            "centroid": (cx, cy),
            "trail":    trail,
            "missing":  0,
        }
        self._assign_color(tid)

    def get_all_trails(self):
        """Returns {track_id: (trail_deque, color)} for drawing."""
        return {
            tid: (tr["trail"], self.color_map.get(tid, (0, 255, 0)))
            for tid, tr in self.tracks.items()
        }

# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY MAP  — accumulates all paths for the full session
# ─────────────────────────────────────────────────────────────────────────────
class TrajectoryMap:
    def __init__(self, width, height):
        self.w        = width
        self.h        = height
        self.canvas   = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_pts = {}   # track_id → last point drawn

    def update(self, tracked_bugs, tracker):
        trails = tracker.get_all_trails()
        for tid, (trail, color) in trails.items():
            pts = list(trail)
            if len(pts) >= 2:
                # draw only new segment (last two points)
                p1 = pts[-2]
                p2 = pts[-1]
                cv2.line(self.canvas, p1, p2, color, 1, cv2.LINE_AA)

    def get_display(self, scale=0.35):
        small = cv2.resize(self.canvas, (int(self.w * scale), int(self.h * scale)))
        # dim background so paths stand out
        bg = np.zeros_like(small)
        bg[:] = (15, 15, 15)
        mask = np.any(small > 0, axis=2)
        bg[mask] = small[mask]
        cv2.putText(bg, "FLIGHT PATHS", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return bg

    def save(self, path):
        cv2.imwrite(path, self.canvas)
        print(f"    Trajectory map → {path}")

# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND SUBTRACTOR + DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def make_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=16, detectShadows=False)

def detect_bugs(frame, bg_subtractor, cfg):
    """Returns (fg_mask, detections) where detections = [(cx,cy,area,x,y,w,h)]"""
    fg = bg_subtractor.apply(frame, learningRate=float(cfg["bg_learning_rate"]))
    _, thresh = cv2.threshold(fg, cfg["motion_threshold"], 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    if cfg["blob_dilation"] > 0:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg["blob_dilation"]*2+1, cfg["blob_dilation"]*2+1))
        cleaned = cv2.dilate(cleaned, dk, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections  = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= cfg["min_bug_area"]:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w / 2
            cy = y + h / 2
            detections.append((cx, cy, area, x, y, w, h))
    return cleaned, detections

def render_motion_frame(mono, tracked_bugs, tracker):
    """Draw bounding boxes and trails onto a BGR frame."""
    out = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    # Draw trails for all active tracks
    trails = tracker.get_all_trails()
    for tid, (trail, color) in trails.items():
        pts = list(trail)
        for k in range(1, len(pts)):
            alpha = k / len(pts)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(out, pts[k-1], pts[k], faded, 1, cv2.LINE_AA)

    # Draw bounding boxes for current detections
    for tid, cx, cy, area, x, y, w, h, color in tracked_bugs:
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 1)
        cv2.putText(out, f"#{tid} {int(area)}px",
                    (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return out

# ─────────────────────────────────────────────────────────────────────────────
#  MENU
# ─────────────────────────────────────────────────────────────────────────────
def clear():
    os.system("clear")

def print_main_menu(cfg, num_cams):
    clear()
    dur    = f"{int(cfg['duration_seconds'])}s" if cfg["duration_seconds"] > 0 else "until S pressed"
    prev   = "ON" if cfg["show_preview"] else "OFF"
    folder = cfg["save_folder"].replace(os.path.expanduser("~"), "~")
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FLIR Mosquito Tracker — Motion + Flight Paths      ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Cameras detected : {num_cams:<33}║")
    print("╠════════════════════ File / Camera ═══════════════════╣")
    print(f"║  [1]  Save folder     : {folder[-30:]:<30}║")
    print(f"║  [2]  Camera 0 name   : {cfg['camera_0_name']:<30}║")
    print(f"║  [3]  Camera 1 name   : {cfg['camera_1_name']:<30}║")
    print(f"║  [4]  Frame rate      : {str(cfg['frame_rate']) + ' fps':<30}║")
    print(f"║  [5]  Exposure time   : {str(cfg['exposure_time']) + ' µs':<30}║")
    print(f"║  [6]  Gain            : {str(cfg['gain_db']) + ' dB':<30}║")
    print(f"║  [7]  Resolution      : {str(cfg['image_width']) + ' x ' + str(cfg['image_height']):<30}║")
    print("╠════════════════════ Motion Detection ════════════════╣")
    print(f"║  [8]  BG learning rate: {cfg['bg_learning_rate']:<30}║")
    print(f"║  [9]  Motion threshold: {str(cfg['motion_threshold']) + '  (0–255)':<30}║")
    print(f"║  [10] Min bug area    : {str(cfg['min_bug_area']) + ' px²':<30}║")
    print(f"║  [11] Blob dilation   : {cfg['blob_dilation']:<30}║")
    print("╠════════════════════ Tracking ════════════════════════╣")
    print(f"║  [12] Trail length    : {str(cfg['trail_length']) + ' frames':<30}║")
    print(f"║  [13] Max match dist  : {str(cfg['max_match_distance']) + ' px':<30}║")
    print(f"║  [14] Track timeout   : {str(cfg['track_timeout']) + ' frames':<30}║")
    print("╠════════════════════ Display ══════════════════════════╣")
    print(f"║  [15] Duration        : {dur:<30}║")
    print(f"║  [16] Preview window  : {prev:<30}║")
    print(f"║  [17] Preview scale   : {cfg['preview_scale']:<30}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  [S]  Save settings    [R]  Start Recording          ║")
    print("║  [Q]  Quit                                           ║")
    print("╚══════════════════════════════════════════════════════╝")

def print_recording_menu(cfg, elapsed, bug_counts):
    clear()
    counts = "  ".join([f"cam{i}: {c} bug{'s' if c!=1 else ''}"
                        for i, c in enumerate(bug_counts)])
    print("╔══════════════════════════════════════════════════════╗")
    print("║            ● RECORDING IN PROGRESS                   ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Elapsed          : {elapsed:.1f}s{'':<27}║")
    print(f"║  Bugs in frame    : {counts:<33}║")
    print("╠════════════════ Adjust While Recording ══════════════╣")
    print(f"║  [E]  Exposure        : {str(cfg['exposure_time']) + ' µs':<30}║")
    print(f"║  [G]  Gain            : {str(cfg['gain_db']) + ' dB':<30}║")
    print(f"║  [T]  Motion threshold: {cfg['motion_threshold']:<30}║")
    print(f"║  [L]  BG learning rate: {cfg['bg_learning_rate']:<30}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  [S]  Stop & Save recording                          ║")
    print("╚══════════════════════════════════════════════════════╝")

# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def configure_camera(cam, cfg, name):
    print(f"    Configuring {name}...")
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(float(cfg["frame_rate"]))
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureTime.SetValue(float(cfg["exposure_time"]))
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(float(cfg["gain_db"]))
    cam.OffsetX.SetValue(0)
    cam.OffsetY.SetValue(0)
    cam.Width.SetValue(int(cfg["image_width"]))
    cam.Height.SetValue(int(cfg["image_height"]))
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

def make_writer(filepath, cfg, color=True):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    w = cv2.VideoWriter(filepath, fourcc, float(cfg["frame_rate"]),
                        (int(cfg["image_width"]), int(cfg["image_height"])),
                        isColor=color)
    if not w.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(filepath, fourcc, float(cfg["frame_rate"]),
                            (int(cfg["image_width"]), int(cfg["image_height"])),
                            isColor=color)
    if not w.isOpened():
        raise RuntimeError(f"Could not open writer: {filepath}")
    return w

def apply_exposure(cameras, value):
    for cam in cameras:
        try: cam.ExposureTime.SetValue(float(value))
        except Exception as e: print(f"  Exposure error: {e}")

def apply_gain(cameras, value):
    for cam in cameras:
        try: cam.Gain.SetValue(float(value))
        except Exception as e: print(f"  Gain error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  CAPTURE THREAD
# ─────────────────────────────────────────────────────────────────────────────
def capture_loop(cam, raw_writer, motion_writer, csv_writer, csv_lock,
                 cam_idx, name, stop_event, preview_queue,
                 bug_count_ref, cfg_ref, traj_map, tracker):
    bg_sub      = make_bg_subtractor()
    frame_count = 0
    t0          = time.time()

    while not stop_event.is_set():
        try:
            img = cam.GetNextImage(1000)
            if img.IsIncomplete():
                img.Release()
                continue

            mono        = img.GetNDArray()
            img.Release()
            frame_count += 1
            timestamp   = time.time() - t0
            cfg         = cfg_ref[0]

            # RAW video
            raw_writer.write(cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR))

            # Detect bugs
            _, detections = detect_bugs(mono, bg_sub, cfg)

            # Track bugs — assign persistent IDs
            tracked = tracker.update(detections)
            bug_count_ref[0] = len(tracked)

            # Update trajectory map
            traj_map.update(tracked, tracker)

            # Render motion frame with trails + boxes
            motion_frame = render_motion_frame(mono, tracked, tracker)
            motion_writer.write(motion_frame)

            # Log to CSV
            if tracked:
                with csv_lock:
                    for tid, cx, cy, area, x, y, w, h, color in tracked:
                        csv_writer.writerow([
                            f"{timestamp:.4f}",
                            frame_count,
                            cam_idx,
                            tid,
                            f"{cx:.1f}",
                            f"{cy:.1f}",
                            int(area),
                            x, y, w, h
                        ])

            # Push to preview
            if preview_queue is not None:
                preview_queue.append(motion_frame.copy())
                if len(preview_queue) > 2:
                    preview_queue.pop(0)

        except PySpin.SpinnakerException as e:
            if not stop_event.is_set():
                print(f"\n  [{name}] Camera error: {e}")
            break
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n  [{name}] Error: {e}")
            break

    elapsed = time.time() - t0
    fps     = frame_count / elapsed if elapsed > 0 else 0
    print(f"  [{name}]  {frame_count} frames  {elapsed:.1f}s  ({fps:.1f} fps)")

# ─────────────────────────────────────────────────────────────────────────────
#  RECORDING SESSION
# ─────────────────────────────────────────────────────────────────────────────
def run_recording(cfg, cam_list):
    num_cams  = min(cam_list.GetSize(), 2)
    cam_names = [cfg["camera_0_name"], cfg["camera_1_name"]]

    os.makedirs(cfg["save_folder"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_paths    = [os.path.join(cfg["save_folder"], f"{cam_names[i]}_{ts}_raw.mp4")
                    for i in range(num_cams)]
    motion_paths = [os.path.join(cfg["save_folder"], f"{cam_names[i]}_{ts}_motion.mp4")
                    for i in range(num_cams)]
    csv_path     =  os.path.join(cfg["save_folder"], f"tracks_{ts}.csv")
    traj_paths   = [os.path.join(cfg["save_folder"], f"{cam_names[i]}_{ts}_trajmap.png")
                    for i in range(num_cams)]

    # Open CSV
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time_s", "frame", "camera", "bug_id",
                         "cx", "cy", "area_px2", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    csv_lock = threading.Lock()

    print("\n  Initialising cameras...")
    cameras, raw_writers, motion_writers, queues = [], [], [], []
    bug_counts  = [[0] for _ in range(num_cams)]
    cfg_refs    = [[dict(cfg)] for _ in range(num_cams)]
    traj_maps   = [TrajectoryMap(cfg["image_width"], cfg["image_height"])
                   for _ in range(num_cams)]
    trackers    = [BugTracker(cfg) for _ in range(num_cams)]

    try:
        for i in range(num_cams):
            cam = cam_list[i]
            cam.Init()
            configure_camera(cam, cfg, cam_names[i])
            cam.BeginAcquisition()
            cameras.append(cam)
            raw_writers.append(make_writer(raw_paths[i], cfg, color=True))
            motion_writers.append(make_writer(motion_paths[i], cfg, color=True))
            queues.append([] if cfg["show_preview"] else None)
            print(f"    RAW    → {raw_paths[i]}")
            print(f"    MOTION → {motion_paths[i]}")
        print(f"    TRACKS → {csv_path}")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        for cam in cameras:
            try: cam.EndAcquisition(); cam.DeInit()
            except Exception: pass
        for w in raw_writers + motion_writers:
            w.release()
        csv_file.close()
        input("\n  Press Enter to return to menu...")
        return

    stop_event = threading.Event()
    threads    = []
    for i, cam in enumerate(cameras):
        t = threading.Thread(
            target=capture_loop,
            args=(cam, raw_writers[i], motion_writers[i],
                  csv_writer, csv_lock, i, cam_names[i],
                  stop_event, queues[i],
                  bug_counts[i], cfg_refs[i],
                  traj_maps[i], trackers[i]),
            daemon=True
        )
        t.start()
        threads.append(t)

    pw           = int(cfg["image_width"]  * cfg["preview_scale"])
    ph           = int(cfg["image_height"] * cfg["preview_scale"])
    traj_scale   = 0.3
    t0           = time.time()
    dur          = cfg["duration_seconds"]
    last_refresh = 0

    print("\n  Recording started. S = stop  E/G/T/L = adjust.\n")

    try:
        while True:
            elapsed = time.time() - t0
            if dur > 0 and elapsed >= dur:
                print(f"\n  Duration limit reached.")
                break

            if elapsed - last_refresh >= 1.0:
                counts = [bc[0] for bc in bug_counts]
                print_recording_menu(cfg, elapsed, counts)
                last_refresh = elapsed

            if cfg["show_preview"]:
                for i, q in enumerate(queues):
                    if q:
                        frame = q[-1]
                        small = cv2.resize(frame, (pw, ph))
                        cv2.putText(small,
                                    f"{cam_names[i]}  {elapsed:.0f}s  bugs:{bug_counts[i][0]}",
                                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow(cam_names[i], small)

                    # Trajectory map window
                    tmap = traj_maps[i].get_display(scale=traj_scale)
                    cv2.imshow(f"{cam_names[i]} — flight paths", tmap)

                cv2.waitKey(1)

            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.readline().strip().upper()

                if key == "S":
                    print("\n  Stopping and saving...")
                    break
                elif key == "E":
                    try:
                        val = float(input("  New exposure (µs): ").strip())
                        apply_exposure(cameras, val)
                        cfg["exposure_time"] = val
                        for cr in cfg_refs: cr[0]["exposure_time"] = val
                    except ValueError: print("  Invalid.")
                elif key == "G":
                    try:
                        val = float(input("  New gain (0–47 dB): ").strip())
                        apply_gain(cameras, val)
                        cfg["gain_db"] = val
                        for cr in cfg_refs: cr[0]["gain_db"] = val
                    except ValueError: print("  Invalid.")
                elif key == "T":
                    try:
                        val = int(input("  New motion threshold (0–255): ").strip())
                        cfg["motion_threshold"] = val
                        for cr in cfg_refs: cr[0]["motion_threshold"] = val
                    except ValueError: print("  Invalid.")
                elif key == "L":
                    try:
                        val = float(input("  New BG learning rate (0.0–1.0): ").strip())
                        cfg["bg_learning_rate"] = val
                        for cr in cfg_refs: cr[0]["bg_learning_rate"] = val
                    except ValueError: print("  Invalid.")

    except KeyboardInterrupt:
        print("\n  Ctrl+C — stopping.")

 # ── Clean shutdown ─────────────────────────────────────────────────────
    print("\n  Stopping threads...")
    stop_event.set()
    for t in threads:
        t.join(timeout=5)

    print("  Releasing cameras...")
    for cam in cameras:
        try: cam.EndAcquisition(); cam.DeInit()
        except Exception as e: print(f"  Camera release warning: {e}")

    print("  Releasing video writers...")
    for w in raw_writers + motion_writers:
        try: w.release()
        except Exception as e: print(f"  Writer release warning: {e}")

    print("  Closing CSV...")
    try:
        csv_file.flush()
        csv_file.close()
        print(f"  CSV saved → {csv_path}")
    except Exception as e:
        print(f"  CSV save error: {e}")

    print("  Saving trajectory maps...")
    for i in range(num_cams):
        try:
            os.makedirs(cfg["save_folder"], exist_ok=True)
            result = cv2.imwrite(traj_paths[i], traj_maps[i].canvas)
            if result:
                print(f"  Trajectory map saved → {traj_paths[i]}")
            else:
                print(f"  WARNING: cv2.imwrite failed for {traj_paths[i]}")
        except Exception as e:
            print(f"  Trajectory map error: {e}")

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("\n  Files saved:")
    all_files = (raw_paths[:num_cams] + motion_paths[:num_cams] +
                 [csv_path] + traj_paths[:num_cams])
    for p in all_files:
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1048576
            print(f"    {os.path.basename(p)}  ({mb:.1f} MB)")
        else:
            print(f"    MISSING: {p}")

    input("\n  Press Enter to return to menu...")

# ─────────────────────────────────────────────────────────────────────────────
#  SETTING EDITOR HELPER
# ─────────────────────────────────────────────────────────────────────────────
def ask(prompt, current, cast=str, validate=None):
    while True:
        raw = input(f"  {prompt} [{current}]: ").strip()
        if raw == "":
            return current
        try:
            val = cast(raw)
            if validate and not validate(val):
                raise ValueError
            return val
        except (ValueError, TypeError):
            print("  Invalid value — try again.")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Install scipy if missing
    if not HAS_SCIPY:
        print("  Note: scipy not found — using greedy blob matching.")
        print("  For better tracking: pip install scipy\n")

    cfg      = load_settings()
    system   = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    try:
        while True:
            num_cams = cam_list.GetSize()
            print_main_menu(cfg, num_cams)
            choice = input("  Enter option: ").strip().upper()

            if   choice == "1":  cfg["save_folder"]        = ask("Save folder", cfg["save_folder"])
            elif choice == "2":  cfg["camera_0_name"]      = ask("Camera 0 name", cfg["camera_0_name"])
            elif choice == "3":  cfg["camera_1_name"]      = ask("Camera 1 name", cfg["camera_1_name"])
            elif choice == "4":  cfg["frame_rate"]         = ask("Frame rate (fps)", cfg["frame_rate"], float, lambda v: 1<=v<=200)
            elif choice == "5":  cfg["exposure_time"]      = ask("Exposure (µs)", cfg["exposure_time"], float, lambda v: v>0)
            elif choice == "6":  cfg["gain_db"]            = ask("Gain (0–47 dB)", cfg["gain_db"], float, lambda v: 0<=v<=47)
            elif choice == "7":
                cfg["image_width"]  = ask("Width  (max 1440)", cfg["image_width"],  int, lambda v: 0<v<=1440)
                cfg["image_height"] = ask("Height (max 1080)", cfg["image_height"], int, lambda v: 0<v<=1080)
            elif choice == "8":  cfg["bg_learning_rate"]  = ask("BG learning rate (0–1)", cfg["bg_learning_rate"], float, lambda v: 0<=v<=1)
            elif choice == "9":  cfg["motion_threshold"]  = ask("Motion threshold (0–255)", cfg["motion_threshold"], int, lambda v: 0<=v<=255)
            elif choice == "10": cfg["min_bug_area"]      = ask("Min bug area (px²)", cfg["min_bug_area"], int, lambda v: v>0)
            elif choice == "11": cfg["blob_dilation"]     = ask("Blob dilation (0–10)", cfg["blob_dilation"], int, lambda v: 0<=v<=10)
            elif choice == "12": cfg["trail_length"]      = ask("Trail length (frames)", cfg["trail_length"], int, lambda v: v>0)
            elif choice == "13": cfg["max_match_distance"]= ask("Max match distance (px)", cfg["max_match_distance"], int, lambda v: v>0)
            elif choice == "14": cfg["track_timeout"]     = ask("Track timeout (frames)", cfg["track_timeout"], int, lambda v: v>0)
            elif choice == "15": cfg["duration_seconds"]  = ask("Duration (0=manual stop)", cfg["duration_seconds"], int, lambda v: v>=0)
            elif choice == "16": cfg["show_preview"]      = not cfg["show_preview"]
            elif choice == "17": cfg["preview_scale"]     = ask("Preview scale (0.1–1.0)", cfg["preview_scale"], float, lambda v: 0.1<=v<=1)
            elif choice == "S":
                save_settings(cfg)
                time.sleep(1)
            elif choice == "R":
                if num_cams == 0:
                    print("\n  No cameras detected.")
                    time.sleep(2)
                else:
                    save_settings(cfg)
                    run_recording(cfg, cam_list)
            elif choice == "Q":
                print("\n  Goodbye!\n")
                break
    finally:
        cam_list.Clear()
        system.ReleaseInstance()

if __name__ == "__main__":
    main()