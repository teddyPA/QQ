"""
QQ_Cameras — FLIR Mosquito Motion Detection & Tracking
=======================================================
Version   : 2.7
Build date: 2026-04-01

GUI: Dear PyGui (GPU-rendered, no Tcl/Tk dependency)
  - Live camera preview per camera
  - Scrollable settings panel with min/max for every field
  - RECORD button (toggle) — saves to ~/Downloads/QQ
  - FOCUS MODE button — sharpness bar + crosshair, no files saved
  - Per-file output toggles (raw, motion, CSV, trajectory map)
  - Wire exclusion: ignore elongated contours (test wires, aspect-ratio filter)
  - Trap attractiveness metrics: ROI zone per camera, visit count,
    dwell time, approach rate, attractiveness index vs control camera;
    periodic metrics CSV export (configurable interval)
  - Reset Cameras: reinitialise cameras without unplug/replug (on startup
    and via UI button)

Install:
    python3.10 -m pip install "numpy<2" opencv-python scipy dearpygui
    PySpin: cd ~/Downloads
            tar -xzf /Applications/Spinnaker/PySpin/spinnaker_python-4.3.0.189-cp310-cp310-macosx_13_0_arm64.tar.gz
            python3.10 -m pip install spinnaker_python-4.3.0.189-cp310-cp310-macosx_13_0_arm64.whl
"""

VERSION    = "2.7"
BUILD_DATE = "2026-04-01"

import math, os, json, threading, time, sys, csv, queue, signal, traceback
from datetime import datetime
from collections import deque

# ── Crash / diagnostic log ─────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE    = os.path.join(_SCRIPT_DIR, "qq_cameras.log")

def log(msg: str):
    """Append a timestamped line to the log file and echo to stdout."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as _lf:
            _lf.write(line + "\n")
    except Exception:
        pass

def _sigabrt_handler(signum, frame):
    log("SIGABRT received — stack trace follows:")
    log("".join(traceback.format_stack(frame)))
    sys.exit(1)

signal.signal(signal.SIGABRT, _sigabrt_handler)

# ── Dear PyGui ─────────────────────────────────────────────────────────────────
try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("ERROR: dearpygui not found.")
    print("  Fix: python3.10 -m pip install dearpygui")
    sys.exit(1)

# ── NumPy ──────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    if int(np.__version__.split(".")[0]) >= 2:
        print("ERROR: NumPy 2.x incompatible with PySpin.")
        print("  Fix: python3.10 -m pip install 'numpy<2' --force-reinstall")
        sys.exit(1)
except ImportError:
    print("ERROR: numpy not found.  python3.10 -m pip install numpy")
    sys.exit(1)

# ── PySpin ─────────────────────────────────────────────────────────────────────
try:
    import PySpin
except ImportError:
    print("ERROR: PySpin not found.")
    print("  cd ~/Downloads")
    print("  tar -xzf /Applications/Spinnaker/PySpin/"
          "spinnaker_python-4.3.0.189-cp310-cp310-macosx_13_0_arm64.tar.gz")
    print("  python3.10 -m pip install "
          "spinnaker_python-4.3.0.189-cp310-cp310-macosx_13_0_arm64.whl")
    sys.exit(1)

# ── OpenCV ─────────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not found.  python3.10 -m pip install opencv-python")
    sys.exit(1)

# ── SciPy (optional) ───────────────────────────────────────────────────────────
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "save_folder":          os.path.expanduser("~/Downloads/QQ"),
    "camera_0_name":        "camera_left",
    "camera_1_name":        "camera_right",
    "frame_rate":           30.0,
    "exposure_time":        8000.0,
    "gain_db":              0.0,
    "image_width":          1440,
    "image_height":         1080,
    "duration_seconds":     0,
    "bg_learning_rate":     0.005,
    "motion_threshold":     20,
    "min_bug_area":         30,
    "blob_dilation":        3,
    "trail_length":         60,
    "max_match_distance":   80,
    "track_timeout":        10,
    "save_raw":             True,
    "save_motion":          True,
    "save_csv":             True,
    "save_traj":            True,
    "wire_aspect_ratio":    4.0,   # contours longer than this × wide are ignored (0=off)
    # ── Trap zone ROI (camera pixel space, default 1440×1080) ──────────────────
    "trap_enabled":         True,
    "trap_cx_0":            720,   # camera 0 trap centre X
    "trap_cy_0":            540,   # camera 0 trap centre Y
    "trap_r_0":             150,   # camera 0 trap radius (px)
    "trap_cx_1":            720,   # camera 1 trap centre X
    "trap_cy_1":            540,   # camera 1 trap centre Y
    "trap_r_1":             150,   # camera 1 trap radius (px)
    "metrics_interval_s":   60,    # seconds between periodic metrics CSV rows
}

# Settings file lives next to the script so it persists across runs
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "camera_settings.json")

SETTING_RANGES = {
    "frame_rate":           (1,      200,    "fps"),
    "exposure_time":        (100,    999999, "µs"),
    "gain_db":              (0.0,    47.0,   "dB"),
    "image_width":          (1,      1440,   "px"),
    "image_height":         (1,      1080,   "px"),
    "bg_learning_rate":     (0.001,  1.0,    ""),
    "motion_threshold":     (0,      255,    ""),
    "min_bug_area":         (1,      9999,   "px²"),
    "blob_dilation":        (0,      10,     "px"),
    "trail_length":         (1,      300,    "frames"),
    "max_match_distance":   (1,      500,    "px"),
    "track_timeout":        (1,      300,    "frames"),
    "duration_seconds":     (0,      86400,  "s (0=manual)"),
    "wire_aspect_ratio":    (0,      20,     "ratio (0=off)"),
    "trap_cx_0":            (0,      1440,   "px"),
    "trap_cy_0":            (0,      1080,   "px"),
    "trap_r_0":             (10,     720,    "px"),
    "trap_cx_1":            (0,      1440,   "px"),
    "trap_cy_1":            (0,      1080,   "px"),
    "trap_r_1":             (10,     720,    "px"),
    "metrics_interval_s":   (10,     3600,   "s"),
}

TRAIL_COLORS = [
    (0,255,0),(0,200,255),(255,100,0),(180,0,255),
    (255,255,0),(0,80,255),(255,0,180),(0,255,150),
    (255,50,50),(150,255,50),(50,150,255),(255,200,50),
]

# Texture dimensions (fixed — controls image quality).
# Display dimensions are computed dynamically in QQCamerasApp and stored as
# self.disp_w / self.disp_h; the DPG image widget scales the texture to fit.
TEX_W, TEX_H = 1200, 900

# Camera hardware settings that can be applied live while cameras are running
_LIVE_CAM_SETTINGS = {"gain_db", "exposure_time", "frame_rate"}

# ──────────────────────────────────────────────────────────────────────────────
#  SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
def load_settings():
    cfg = dict(DEFAULTS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def save_settings(cfg):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
#  TRAP STATISTICS  (thread-safe accumulator)
# ──────────────────────────────────────────────────────────────────────────────
class TrapStats:
    """Per-camera trap-zone statistics. Updated by capture thread; read by main thread."""

    def __init__(self):
        self._lock            = threading.Lock()
        self.visit_count      = 0       # unique bug IDs that entered zone
        self._visit_ids       = set()   # IDs ever inside zone
        self._in_zone         = {}      # {tid: frames_inside_so_far}
        self.dwell_frames     = []      # completed dwell times (in frames)
        self.approach_count   = 0       # frames a bug moved closer to centre
        self.away_count       = 0       # frames a bug moved farther from centre
        self.total_detections = 0       # cumulative bug-frame detections
        self._prev_dist       = {}      # {tid: distance to centre previous frame}

    def reset(self):
        with self._lock:
            self.__init__()

    def update(self, tracked, cx, cy, r):
        """Call once per frame from the capture thread.
        tracked = list of (tid, bx, by, area, x, y, w, h, color).
        cx, cy, r  — trap zone centre and radius in camera pixel space.
        """
        with self._lock:
            current_ids = set()
            for item in tracked:
                tid, bx, by = item[0], item[1], item[2]
                self.total_detections += 1
                dist = math.hypot(bx - cx, by - cy)
                current_ids.add(tid)

                if dist <= r:
                    if tid not in self._visit_ids:
                        self._visit_ids.add(tid)
                        self.visit_count += 1
                    self._in_zone[tid] = self._in_zone.get(tid, 0) + 1
                else:
                    if tid in self._in_zone:
                        self.dwell_frames.append(self._in_zone.pop(tid))

                if tid in self._prev_dist:
                    prev = self._prev_dist[tid]
                    if dist < prev - 0.5:
                        self.approach_count += 1
                    elif dist > prev + 0.5:
                        self.away_count += 1
                self._prev_dist[tid] = dist

            # Clean up tracks that disappeared
            for tid in list(self._in_zone):
                if tid not in current_ids:
                    self.dwell_frames.append(self._in_zone.pop(tid))
            for tid in list(self._prev_dist):
                if tid not in current_ids:
                    del self._prev_dist[tid]

    def snapshot(self, frame_rate):
        """Return a dict of current metrics (safe to call from main thread)."""
        with self._lock:
            all_d    = self.dwell_frames + list(self._in_zone.values())
            mean_d_s = (sum(all_d) / len(all_d) / max(frame_rate, 1)) if all_d else 0.0
            total    = max(self.total_detections, 1)
            return {
                "visit_count":    self.visit_count,
                "mean_dwell_s":   mean_d_s,
                "approach_count": self.approach_count,
                "away_count":     self.away_count,
                "tz_ratio":       self.visit_count / total,
            }

# ──────────────────────────────────────────────────────────────────────────────
#  BUG TRACKER
# ──────────────────────────────────────────────────────────────────────────────
class BugTracker:
    def __init__(self, cfg):
        self.cfg        = cfg
        self.next_id    = 0
        self.tracks     = {}
        self.color_map  = {}
        self.color_pool = list(range(len(TRAIL_COLORS)))

    def _assign_color(self, tid):
        idx = self.color_pool.pop(0) if self.color_pool else tid % len(TRAIL_COLORS)
        self.color_map[tid] = TRAIL_COLORS[idx]

    def _release_color(self, tid):
        if tid in self.color_map:
            idx = TRAIL_COLORS.index(self.color_map[tid])
            if idx not in self.color_pool:
                self.color_pool.insert(0, idx)
            del self.color_map[tid]

    def update(self, detections):
        max_dist   = self.cfg["max_match_distance"]
        trail_len  = self.cfg["trail_length"]
        timeout    = self.cfg["track_timeout"]
        active_ids = list(self.tracks.keys())

        if active_ids and detections:
            cost = np.zeros((len(active_ids), len(detections)))
            for i, tid in enumerate(active_ids):
                tx, ty = self.tracks[tid]["centroid"]
                for j, (cx, cy, *_) in enumerate(detections):
                    cost[i, j] = np.hypot(tx-cx, ty-cy)
            if HAS_SCIPY:
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                row_ind, col_ind, used = [], [], set()
                for i in range(len(active_ids)):
                    bj = min((j for j in range(len(detections)) if j not in used),
                             key=lambda j: cost[i,j], default=None)
                    if bj is not None:
                        row_ind.append(i); col_ind.append(bj); used.add(bj)
            mt, md = set(), set()
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= max_dist:
                    tid = active_ids[i]
                    cx, cy, *_ = detections[j]
                    tr = self.tracks[tid]["trail"]
                    tr.append((int(cx), int(cy)))
                    if len(tr) > trail_len: tr.popleft()
                    self.tracks[tid]["centroid"] = (cx, cy)
                    self.tracks[tid]["missing"]  = 0
                    mt.add(i); md.add(j)
            for i, tid in enumerate(active_ids):
                if i not in mt: self.tracks[tid]["missing"] += 1
            for j, det in enumerate(detections):
                if j not in md: self._new_track(det)
        else:
            for det in detections: self._new_track(det)
            for tid in active_ids:  self.tracks[tid]["missing"] += 1

        for tid in list(self.tracks):
            if self.tracks[tid]["missing"] > timeout:
                self._release_color(tid); del self.tracks[tid]

        results = []
        for tid, tr in self.tracks.items():
            if tr["missing"] == 0:
                cx, cy = tr["centroid"]
                for det in detections:
                    dcx, dcy, area, x, y, w, h = det
                    if abs(dcx-cx) < 2 and abs(dcy-cy) < 2:
                        results.append((tid,cx,cy,area,x,y,w,h,
                                        self.color_map.get(tid,(0,255,0))))
                        break
        return results

    def _new_track(self, det):
        cx, cy, *_ = det
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = {
            "centroid": (cx, cy),
            "trail":    deque([(int(cx),int(cy))], maxlen=self.cfg["trail_length"]),
            "missing":  0,
        }
        self._assign_color(tid)

    def get_all_trails(self):
        return {tid: (tr["trail"], self.color_map.get(tid,(0,255,0)))
                for tid, tr in self.tracks.items()}

# ──────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY MAP
# ──────────────────────────────────────────────────────────────────────────────
class TrajectoryMap:
    def __init__(self, w, h):
        self.w = w; self.h = h
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

    def update(self, tracked, tracker):
        for tid, (trail, color) in tracker.get_all_trails().items():
            pts = list(trail)
            if len(pts) >= 2:
                cv2.line(self.canvas, pts[-2], pts[-1], color, 1, cv2.LINE_AA)

    def save(self, path):
        cv2.imwrite(path, self.canvas)

# ──────────────────────────────────────────────────────────────────────────────
#  DETECTION / RENDERING
# ──────────────────────────────────────────────────────────────────────────────
def make_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=16, detectShadows=False)

def detect_bugs(frame, bg_sub, cfg):
    fg        = bg_sub.apply(frame, learningRate=float(cfg["bg_learning_rate"]))
    _, thresh = cv2.threshold(fg, cfg["motion_threshold"], 255, cv2.THRESH_BINARY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    if cfg["blob_dilation"] > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
             (cfg["blob_dilation"]*2+1,)*2)
        cleaned = cv2.dilate(cleaned, dk, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wire_ratio  = float(cfg.get("wire_aspect_ratio", 0))
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["min_bug_area"]:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Wire exclusion: real bugs are roughly round/blobby;
        # wires are long thin shapes with high aspect ratio
        if wire_ratio > 0:
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > wire_ratio:
                continue
        dets.append((x+w/2, y+h/2, area, x, y, w, h))
    return cleaned, dets

def render_motion_frame(mono, tracked, tracker, trap_zone=None):
    """Render motion frame with trails, bounding boxes, and optional trap zone circle."""
    out = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    for tid, (trail, color) in tracker.get_all_trails().items():
        pts = list(trail)
        for k in range(1, len(pts)):
            alpha = k / len(pts)
            cv2.line(out, pts[k-1], pts[k],
                     tuple(int(c*alpha) for c in color), 1, cv2.LINE_AA)
    for tid,cx,cy,area,x,y,w,h,color in tracked:
        cv2.rectangle(out,(x,y),(x+w,y+h),color,1)
        cv2.putText(out,f"#{tid} {int(area)}px",(x,y-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,color,1)
    # Trap zone overlay
    if trap_zone is not None:
        tzx, tzy, tzr = int(trap_zone[0]), int(trap_zone[1]), int(trap_zone[2])
        cv2.circle(out, (tzx, tzy), tzr, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(out, (tzx, tzy), (0, 200, 255),
                       cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    return out

# ──────────────────────────────────────────────────────────────────────────────
#  CAMERA HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def configure_camera(cam, cfg, name):
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(float(cfg["frame_rate"]))
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureTime.SetValue(float(cfg["exposure_time"]))
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(float(cfg["gain_db"]))
    cam.OffsetX.SetValue(0); cam.OffsetY.SetValue(0)
    cam.Width.SetValue(int(cfg["image_width"]))
    cam.Height.SetValue(int(cfg["image_height"]))
    cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

def make_writer(filepath, cfg):
    for fourcc_str in ("avc1", "mp4v"):
        w = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*fourcc_str),
                            float(cfg["frame_rate"]),
                            (int(cfg["image_width"]), int(cfg["image_height"])),
                            isColor=True)
        if w.isOpened():
            return w
    raise RuntimeError(f"Could not open video writer: {filepath}")

def push_frame(q, frame):
    if q is None: return
    try: q.put_nowait(frame)
    except queue.Full:
        try: q.get_nowait()
        except queue.Empty: pass
        try: q.put_nowait(frame)
        except Exception: pass

# ──────────────────────────────────────────────────────────────────────────────
#  CAPTURE THREAD
# ──────────────────────────────────────────────────────────────────────────────
def capture_loop(cam, raw_writer, motion_writer, csv_writer, csv_lock,
                 cam_idx, name, stop_event, frame_q,
                 bug_count_ref, cfg_ref, traj_map, tracker, save_opts,
                 trap_stats):
    bg_sub      = make_bg_subtractor()
    frame_count = 0
    t0          = time.time()
    while not stop_event.is_set():
        # Apply any pending hardware setting changes (queued by main thread via cfg_ref[1])
        if cfg_ref[1]:
            pending, cfg_ref[1] = cfg_ref[1].copy(), {}
            for k, v in pending.items():
                try:
                    if k == "gain_db":
                        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                        cam.Gain.SetValue(float(v))
                    elif k == "exposure_time":
                        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                        cam.ExposureTime.SetValue(float(v))
                    elif k == "frame_rate":
                        cam.AcquisitionFrameRateEnable.SetValue(True)
                        cam.AcquisitionFrameRate.SetValue(float(v))
                except Exception as e:
                    log(f"  [{name}] FAILED setting {k}={v}: {e}")
        try:
            img = cam.GetNextImage(1000)
            if img.IsIncomplete(): img.Release(); continue
            mono        = img.GetNDArray(); img.Release()
            frame_count += 1
            ts          = time.time() - t0
            cfg         = cfg_ref[0]
            if save_opts["save_raw"] and raw_writer:
                raw_writer.write(cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR))
            _, dets  = detect_bugs(mono, bg_sub, cfg)
            tracked  = tracker.update(dets)
            bug_count_ref[0] = len(tracked)
            if save_opts["save_traj"]: traj_map.update(tracked, tracker)

            # Trap zone metrics + overlay
            trap_zone = None
            if cfg.get("trap_enabled", False):
                tzx = float(cfg.get(f"trap_cx_{cam_idx}", 720))
                tzy = float(cfg.get(f"trap_cy_{cam_idx}", 540))
                tzr = float(cfg.get(f"trap_r_{cam_idx}",  150))
                trap_stats.update(tracked, tzx, tzy, tzr)
                trap_zone = (tzx, tzy, tzr)

            motion_frame = render_motion_frame(mono, tracked, tracker, trap_zone)
            if save_opts["save_motion"] and motion_writer:
                motion_writer.write(motion_frame)
            if save_opts["save_csv"] and csv_writer and tracked:
                with csv_lock:
                    for tid,cx,cy,area,x,y,w,h,_ in tracked:
                        csv_writer.writerow([f"{ts:.4f}",frame_count,cam_idx,tid,
                                             f"{cx:.1f}",f"{cy:.1f}",int(area),x,y,w,h])
            push_frame(frame_q, motion_frame)
        except PySpin.SpinnakerException as e:
            if not stop_event.is_set(): print(f"  [{name}] Camera error: {e}")
            break
        except Exception as e:
            if not stop_event.is_set(): print(f"  [{name}] Error: {e}")
            break

# ──────────────────────────────────────────────────────────────────────────────
#  FOCUS THREAD
# ──────────────────────────────────────────────────────────────────────────────
def focus_loop(cam, name, stop_event, frame_q, cfg_ref):
    while not stop_event.is_set():
        # Apply any pending hardware setting changes (queued by main thread via cfg_ref[1])
        if cfg_ref[1]:
            pending, cfg_ref[1] = cfg_ref[1].copy(), {}
            for k, v in pending.items():
                try:
                    if k == "gain_db":
                        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                        cam.Gain.SetValue(float(v))
                    elif k == "exposure_time":
                        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                        cam.ExposureTime.SetValue(float(v))
                    elif k == "frame_rate":
                        cam.AcquisitionFrameRateEnable.SetValue(True)
                        cam.AcquisitionFrameRate.SetValue(float(v))
                except Exception as e:
                    log(f"  [{name}] FAILED setting {k}={v}: {e}")
        try:
            img = cam.GetNextImage(1000)
            if img.IsIncomplete(): img.Release(); continue
            mono = img.GetNDArray(); img.Release()
            bgr  = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
            h, w = bgr.shape[:2]
            sharpness = cv2.Laplacian(mono, cv2.CV_64F).var()
            bar_w = max(0, min(int(sharpness / 500 * (w-20)), w-20))
            color = (0,255,0) if sharpness>200 else (0,165,255) if sharpness>80 else (0,60,200)
            cv2.rectangle(bgr,(10,10),(10+bar_w,26),color,-1)
            cv2.rectangle(bgr,(10,10),(w-10,26),(80,80,80),1)
            cv2.putText(bgr,f"Sharpness: {sharpness:.0f}",(14,23),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
            cx, cy = w//2, h//2
            cv2.line(bgr,(cx-30,cy),(cx+30,cy),(0,220,255),1)
            cv2.line(bgr,(cx,cy-30),(cx,cy+30),(0,220,255),1)
            cv2.circle(bgr,(cx,cy),40,(0,220,255),1)
            cv2.putText(bgr,f"FOCUS - {name}  |  NO FILES SAVED",
                        (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,200,255),1)
            push_frame(frame_q, bgr)
        except PySpin.SpinnakerException as e:
            if not stop_event.is_set(): print(f"  [{name}] Focus error: {e}")
            break
        except Exception as e:
            if not stop_event.is_set(): print(f"  [{name}] Error: {e}")
            break

# ──────────────────────────────────────────────────────────────────────────────
#  FRAME → DEAR PYGUI TEXTURE
# ──────────────────────────────────────────────────────────────────────────────
def frame_to_texture(frame_bgr):
    """Convert a BGR OpenCV frame to a flat RGBA float32 numpy array for Dear PyGui."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (TEX_W, TEX_H))
    rgba    = np.ones((TEX_H, TEX_W, 4), dtype=np.float32)
    rgba[:,:,:3] = resized.astype(np.float32) / 255.0
    return rgba.flatten()   # numpy array — DPG accepts it directly (faster than tolist)

def blank_texture():
    """Dark placeholder texture (TEX_W × TEX_H)."""
    data = np.full((TEX_H, TEX_W, 4), 0.06, dtype=np.float32)
    data[:,:,3] = 1.0
    return data.flatten()

# ──────────────────────────────────────────────────────────────────────────────
#  GUI  APPLICATION  (Dear PyGui)
# ──────────────────────────────────────────────────────────────────────────────
class QQCamerasApp:

    def __init__(self):
        self.cfg  = load_settings()
        self.mode = "idle"   # idle | recording | focus

        # PySpin
        try:
            self.system   = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            self.num_cams = min(self.cam_list.GetSize(), 2)
        except Exception:
            self.system = self.cam_list = None
            self.num_cams = 0

        # Reset cameras on startup — clears stale SDK state from any previous crash
        self._reset_cameras()

        # Runtime
        self.cameras        = []
        self.stop_event     = threading.Event()
        self.threads        = []
        self.frame_qs       = [queue.Queue(maxsize=3), queue.Queue(maxsize=3)]
        self.bug_count_refs = [[0], [0]]
        # cfg_refs[i][0] = current motion/detection config dict (read by threads)
        # cfg_refs[i][1] = pending hardware settings dict (applied by threads between frames)
        self.cfg_refs       = [[dict(self.cfg), {}], [dict(self.cfg), {}]]
        self.traj_maps      = [None, None]
        self.trackers       = [None, None]
        self.raw_writers    = [None, None]
        self.motion_writers = [None, None]
        self.csv_file       = None
        self.record_t0      = None
        self.traj_paths     = []
        self.record_save_opts = {}
        self._pending_alert  = None   # set by background cleanup threads

        # Trap attractiveness metrics
        self.trap_stats          = [TrapStats(), TrapStats()]
        self.metrics_csv_file    = None
        self.metrics_csv_writer  = None
        self.metrics_csv_lock    = threading.Lock()
        self.last_metrics_export = 0.0

        # Display dimensions (updated dynamically by viewport resize callback)
        self.disp_w = 400
        self.disp_h = 300

    # ── Camera reset ───────────────────────────────────────────────────────────

    def _reset_cameras(self):
        """Force-reinitialise all cameras to clear stale SDK state from a crash.
        Safe to call even when cameras are in a clean state.
        """
        log("_reset_cameras: starting")
        if self.cam_list is None:
            log("_reset_cameras: no cam_list, skipping")
            return
        n = min(self.cam_list.GetSize(), 2)
        for i in range(n):
            try:
                cam = self.cam_list[i]
                # Attempt to stop any stale acquisition
                try:
                    cam.EndAcquisition()
                    log(f"  cam {i}: EndAcquisition OK (was left acquiring)")
                except Exception:
                    pass  # not acquiring — expected on clean start
                # Init / DeInit cycle clears SDK internal state
                try:
                    cam.Init()
                    log(f"  cam {i}: Init OK")
                except Exception:
                    # Already init'd from a crashed process — DeInit first
                    try:
                        cam.DeInit()
                        cam.Init()
                        log(f"  cam {i}: DeInit+Init OK")
                    except Exception as e2:
                        log(f"  cam {i}: Init failed: {e2}")
                        continue
                try:
                    cam.DeInit()
                    log(f"  cam {i}: DeInit OK")
                except Exception as e:
                    log(f"  cam {i}: DeInit failed: {e}")
            except Exception as e:
                log(f"  cam {i}: reset error: {e}")
        log("_reset_cameras: done")

    def _reset_cameras_cb(self):
        """UI button callback — reset cameras and re-enumerate."""
        if self.mode in ("recording", "focus"):
            self._alert("Stop recording / focus mode first before resetting cameras.")
            return
        log("Manual camera reset triggered from UI")
        self._reset_cameras()
        # Re-enumerate camera list
        if self.cam_list:
            try: self.cam_list.Clear()
            except Exception: pass
        if self.system:
            try:
                self.cam_list = self.system.GetCameras()
                self.num_cams = min(self.cam_list.GetSize(), 2)
                log(f"Re-enumerate: {self.num_cams} camera(s) found")
            except Exception as e:
                log(f"Re-enumerate failed: {e}")
                self.num_cams = 0
        if dpg.does_item_exist("cam_count_text"):
            dpg.set_value("cam_count_text", f"Cameras detected: {self.num_cams}")
        self._alert(f"Cameras reset.\n{self.num_cams} camera(s) found.", title="Reset Done")

    # ── Metrics CSV helpers ────────────────────────────────────────────────────

    def _export_metrics_row(self, writer=None, lock=None):
        """Write one row to the metrics CSV (current accumulated stats)."""
        w = writer or self.metrics_csv_writer
        lk = lock  or self.metrics_csv_lock
        if w is None or self.record_t0 is None:
            return
        elapsed = time.time() - self.record_t0
        fr      = float(self.cfg.get("frame_rate", 30))
        snaps   = [self.trap_stats[i].snapshot(fr)
                   for i in range(min(self.num_cams, 2))]
        s0 = snaps[0] if len(snaps) > 0 else {}
        s1 = snaps[1] if len(snaps) > 1 else {}
        ai = (s0.get("visit_count", 0) / max(s1.get("visit_count", 0), 1)
              if s1 else float("nan"))
        rd = (s0.get("mean_dwell_s", 0) / max(s1.get("mean_dwell_s", 0.001), 0.001)
              if s1 else float("nan"))
        try:
            with lk:
                w.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{elapsed:.1f}",
                    s0.get("visit_count", 0),
                    f"{s0.get('mean_dwell_s', 0):.3f}",
                    f"{s0.get('tz_ratio', 0):.4f}",
                    s0.get("approach_count", 0),
                    s0.get("away_count", 0),
                    s1.get("visit_count", 0),
                    f"{s1.get('mean_dwell_s', 0):.3f}",
                    f"{s1.get('tz_ratio', 0):.4f}",
                    s1.get("approach_count", 0),
                    s1.get("away_count", 0),
                    f"{ai:.4f}" if not math.isnan(ai) else "N/A",
                    f"{rd:.4f}" if not math.isnan(rd) else "N/A",
                ])
                if self.metrics_csv_file:
                    self.metrics_csv_file.flush()
        except Exception as e:
            log(f"metrics export failed: {e}")

    # ── Dear PyGui setup ───────────────────────────────────────────────────────

    def run(self):
        log(f"QQ_Cameras v{VERSION} starting")
        dpg.create_context()
        self._apply_theme()

        # Show & maximise FIRST so we know the actual screen size before building UI
        dpg.create_viewport(
            title=f"QQ_Cameras  v{VERSION}  |  {BUILD_DATE}",
            width=1400, height=900,
            min_width=900, min_height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()
        # Pump a couple of frames so macOS actually applies the maximize
        for _ in range(3):
            dpg.render_dearpygui_frame()

        # Compute initial display sizes and store as instance vars
        self.disp_w, self.disp_h = 400, 300   # safe fallback
        self._update_preview_sizes(dpg.get_viewport_width(), dpg.get_viewport_height())
        log(f"Viewport {dpg.get_viewport_width()}×{dpg.get_viewport_height()}, "
            f"preview {self.disp_w}×{self.disp_h}")

        # Textures are fixed at TEX_W×TEX_H for quality;
        # the add_image widget scales them to disp_w×disp_h
        with dpg.texture_registry():
            self.tex_ids = [
                dpg.add_dynamic_texture(TEX_W, TEX_H, blank_texture()),
                dpg.add_dynamic_texture(TEX_W, TEX_H, blank_texture()),
            ]

        self._build_ui()
        dpg.set_primary_window("main_win", True)
        dpg.set_viewport_resize_callback(self._on_resize)

        # Main render loop
        while dpg.is_dearpygui_running():
            try:
                self._tick()
            except Exception as e:
                log(f"[tick error] {e}\n{traceback.format_exc()}")
            dpg.render_dearpygui_frame()

        self._shutdown()
        dpg.destroy_context()
        log("QQ_Cameras exited cleanly")

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       (22, 22, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        (18, 28, 52))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,        (18, 28, 52))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (12, 22, 44))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (20, 40, 70))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  (30, 60, 100))
                dpg.add_theme_color(dpg.mvThemeCol_Text,           (220, 220, 228))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,   (100, 100, 120))
                dpg.add_theme_color(dpg.mvThemeCol_Button,         (40, 100, 180))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (60, 130, 210))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (80, 160, 240))
                dpg.add_theme_color(dpg.mvThemeCol_Header,         (30, 80, 140))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (40, 100, 160))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,    (12, 22, 44))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,  (40, 70, 120))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,        (14, 20, 38))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  (20, 34, 64))
                dpg.add_theme_color(dpg.mvThemeCol_Separator,      (40, 50, 80))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,    6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,     4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,     4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,      4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,       8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    10, 10)
        dpg.bind_theme(t)

    # ── Viewport resize ────────────────────────────────────────────────────────

    def _on_resize(self, *_):
        self._update_preview_sizes(dpg.get_viewport_width(), dpg.get_viewport_height())

    def _update_preview_sizes(self, vp_w, vp_h):
        """Recompute the largest side-by-side preview that fits the viewport."""
        SETTINGS_W = 320   # settings child-window width
        CHROME_H   = 220   # header + separator + status + buttons + metrics + padding
        SPACING    = 60    # inter-panel + internal padding
        avail_w    = max(400, vp_w - SETTINGS_W - SPACING)
        pw = avail_w // 2
        ph = int(pw * 1080 / 1440)          # 4∶3 — matches FLIR 1440×1080
        max_ph = max(200, vp_h - CHROME_H)
        if ph > max_ph:
            ph = max_ph
            pw = int(ph * 1440 / 1080)
        self.disp_w = max(pw, 200)
        self.disp_h = max(ph, 150)
        # If the UI has been built, update the image widget sizes immediately
        for i in range(2):
            tag = f"cam_img_{i}"
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=self.disp_w, height=self.disp_h)

    # ── UI layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        names = [self.cfg.get("camera_0_name","camera_left"),
                 self.cfg.get("camera_1_name","camera_right")]

        with dpg.window(tag="main_win", no_title_bar=True,
                        no_move=True, no_resize=True):

            # ── Header ────────────────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text(f"QQ_Cameras", color=(79,195,247))
                dpg.add_text(f"  v{VERSION}   {BUILD_DATE}",
                             color=(100,100,130))
                dpg.add_spacer(width=40)
                dpg.add_text("", tag="mode_text", color=(130,130,150))
                dpg.add_spacer(width=20)
                dpg.add_text("", tag="elapsed_text", color=(200,200,210))
                dpg.add_spacer(width=40)
                dpg.add_text(f"Cameras detected: {self.num_cams}",
                             tag="cam_count_text",
                             color=(100,100,130))
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # ── Body: settings | cameras ───────────────────────────────────────
            with dpg.group(horizontal=True):

                # ── Settings panel ────────────────────────────────────────────
                with dpg.child_window(tag="settings_panel", width=300,
                                      border=True, horizontal_scrollbar=False):
                    self._build_settings()

                dpg.add_spacer(width=8)

                # ── Right: previews + metrics + status + buttons ───────────────
                with dpg.group():

                    # Camera previews
                    with dpg.group(horizontal=True):
                        for i in range(2):
                            with dpg.group():
                                dpg.add_text(names[i].upper(),
                                             color=(79,195,247))
                                dpg.add_image(self.tex_ids[i],
                                              tag=f"cam_img_{i}",
                                              width=self.disp_w, height=self.disp_h)
                                dpg.add_text("--",
                                             tag=f"bug_count_{i}",
                                             color=(100,100,130))
                            if i == 0:
                                dpg.add_spacer(width=12)

                    dpg.add_spacer(height=6)

                    # ── Trap Attractiveness Metrics panel ─────────────────────
                    with dpg.group(tag="trap_stats_panel"):
                        with dpg.group(horizontal=True):
                            dpg.add_text("TRAP METRICS",
                                         color=(0, 200, 255))
                            dpg.add_spacer(width=10)
                            dpg.add_text("(live during recording)",
                                         color=(60,60,90))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Cam 0 (treatment):",
                                         color=(100,220,100))
                            dpg.add_spacer(width=6)
                            dpg.add_text("--", tag="trap_stats_0",
                                         color=(200,220,200))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Cam 1 (control):  ",
                                         color=(100,150,255))
                            dpg.add_spacer(width=6)
                            dpg.add_text("--", tag="trap_stats_1",
                                         color=(200,210,255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Index:            ",
                                         color=(220,190,80))
                            dpg.add_spacer(width=6)
                            dpg.add_text("--", tag="trap_attractiveness",
                                         color=(240,220,100))

                    dpg.add_spacer(height=4)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)

                    # Focus hint
                    dpg.add_text("", tag="focus_hint",
                                 color=(255,167,38), wrap=880)
                    dpg.add_spacer(height=2)

                    # Output folder label
                    dpg.add_text(
                        f"Output -> {self.cfg['save_folder']}",
                        tag="folder_label", color=(70,70,100),
                        wrap=880)
                    dpg.add_spacer(height=8)

                    # ── Buttons ───────────────────────────────────────────────
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="  RECORD  ",
                            tag="record_btn",
                            height=46,
                            callback=self._toggle_record)
                        self._btn_color("record_btn", (180,40,40))

                        dpg.add_spacer(width=6)

                        dpg.add_button(
                            label="  FOCUS MODE  ",
                            tag="focus_btn",
                            height=46,
                            callback=self._toggle_focus)
                        self._btn_color("focus_btn", (180,100,20))

                        dpg.add_spacer(width=6)

                        dpg.add_button(
                            label="  Reset Cameras  ",
                            tag="reset_cam_btn",
                            height=46,
                            callback=self._reset_cameras_cb)
                        self._btn_color("reset_cam_btn", (20,100,80))

                        dpg.add_spacer(width=6)

                        dpg.add_button(
                            label="  Save Settings  ",
                            height=46,
                            callback=self._save_settings_cb)
                        self._btn_color(dpg.last_item(), (30,90,160))

                        dpg.add_spacer(width=6)

                        dpg.add_button(
                            label="  Quit  ",
                            height=46,
                            callback=self._quit)
                        self._btn_color(dpg.last_item(), (60,60,80))

    def _btn_color(self, tag, rgb):
        r,g,b = rgb
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (r,g,b,255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (min(r+30,255),min(g+30,255),min(b+30,255),255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (min(r+50,255),min(g+50,255),min(b+50,255),255))
        dpg.bind_item_theme(tag, t)

    # ── Settings panel ─────────────────────────────────────────────────────────

    def _section(self, title):
        dpg.add_spacer(height=6)
        dpg.add_text(title, color=(79,195,247))
        dpg.add_separator()

    def _float_row(self, key, label):
        lo, hi, unit = SETTING_RANGES.get(key, (0.0, 1e9, ""))
        hint = f"  {lo} – {hi}" + (f" {unit}" if unit else "")
        val  = float(self.cfg.get(key, lo))
        dpg.add_input_double(
            label=f"{label}{hint}", tag=f"s_{key}",
            default_value=val,
            min_value=float(lo), max_value=float(hi),
            min_clamped=True, max_clamped=True,
            width=110,
            callback=lambda s, a, k=key: self._cfg_set(k, a))

    def _int_row(self, key, label):
        lo, hi, unit = SETTING_RANGES.get(key, (0, 9999, ""))
        hint = f"  {lo} – {hi}" + (f" {unit}" if unit else "")
        val  = int(self.cfg.get(key, lo))
        dpg.add_input_int(
            label=f"{label}{hint}", tag=f"s_{key}",
            default_value=val,
            min_value=int(lo), max_value=int(hi),
            min_clamped=True, max_clamped=True,
            width=110,
            callback=lambda s, a, k=key: self._cfg_set(k, a))

    def _text_row(self, key, label):
        dpg.add_input_text(
            label=label, tag=f"s_{key}",
            default_value=str(self.cfg.get(key, "")),
            width=150,
            on_enter=True,
            callback=lambda s, a, k=key: self._cfg_set(k, a))

    def _bool_row(self, key, label):
        dpg.add_checkbox(
            label=label, tag=f"s_{key}",
            default_value=bool(self.cfg.get(key, False)),
            callback=lambda s, a, k=key: self._cfg_set(k, a))

    def _build_settings(self):
        self._section("File / Camera")
        self._text_row("save_folder",   "Save folder")
        self._text_row("camera_0_name", "Camera 0 name")
        self._text_row("camera_1_name", "Camera 1 name")
        self._float_row("frame_rate",   "Frame rate")
        self._float_row("exposure_time","Exposure time")
        self._float_row("gain_db",      "Gain")
        self._int_row("image_width",    "Width")
        self._int_row("image_height",   "Height")

        self._section("Motion Detection")
        self._float_row("bg_learning_rate", "BG learn rate")
        self._int_row("motion_threshold",   "Motion thresh")
        self._int_row("min_bug_area",        "Min bug area")
        self._int_row("blob_dilation",       "Blob dilation")
        self._float_row("wire_aspect_ratio", "Wire filter ratio")

        self._section("Tracking")
        self._int_row("trail_length",       "Trail length")
        self._int_row("max_match_distance", "Max match dist")
        self._int_row("track_timeout",      "Track timeout")

        self._section("Display / Duration")
        self._int_row("duration_seconds",   "Duration (s)")

        self._section("Output Files")
        self._bool_row("save_raw",    "Raw video")
        self._bool_row("save_motion", "Motion video")
        self._bool_row("save_csv",    "CSV tracks")
        self._bool_row("save_traj",   "Traj map")

        self._section("Trap Attractiveness Metrics")
        self._bool_row("trap_enabled", "Enable trap metrics")
        dpg.add_text("  Cam 0 (treatment trap zone):", color=(100,220,100))
        self._int_row("trap_cx_0", "  Centre X")
        self._int_row("trap_cy_0", "  Centre Y")
        self._int_row("trap_r_0",  "  Radius")
        dpg.add_text("  Cam 1 (control trap zone):", color=(100,150,255))
        self._int_row("trap_cx_1", "  Centre X")
        self._int_row("trap_cy_1", "  Centre Y")
        self._int_row("trap_r_1",  "  Radius")
        self._int_row("metrics_interval_s", "Export interval")

    # ── Config helpers ─────────────────────────────────────────────────────────

    def _cfg_set(self, key, value):
        self.cfg[key] = value
        if key == "save_folder":
            dpg.set_value("folder_label", f"Output -> {value}")
        for cr in self.cfg_refs:
            cr[0] = dict(self.cfg)
            # Queue hardware setting changes; the capture/focus thread applies
            # them between frames (thread-safe — same thread that owns the camera)
            if key in _LIVE_CAM_SETTINGS:
                cr[1][key] = value
                log(f"Queued live cam setting: {key}={value}  "
                    f"(cameras running: {len(self.cameras) > 0})")
        # Auto-save every time a setting changes (no explicit Save button required)
        try:
            save_settings(self.cfg)
        except Exception as e:
            log(f"Auto-save failed: {e}")

    # ── Per-tick update ────────────────────────────────────────────────────────

    def _tick(self):
        # Push camera frames to textures
        for i in range(min(self.num_cams, 2)):
            try:
                frame = self.frame_qs[i].get_nowait()
                dpg.set_value(self.tex_ids[i], frame_to_texture(frame))
            except queue.Empty:
                pass

        # Show deferred alerts from background cleanup threads
        if self._pending_alert:
            msg, title = self._pending_alert
            self._pending_alert = None
            self._alert(msg, title=title)

        # Update recording status
        if self.mode == "recording" and self.record_t0:
            elapsed = time.time() - self.record_t0
            m, s = int(elapsed//60), int(elapsed%60)
            dpg.set_value("elapsed_text", f"{m:02d}:{s:02d}")
            for i in range(self.num_cams):
                count = self.bug_count_refs[i][0]
                dpg.set_value(f"bug_count_{i}",
                              f"{count} bug{'s' if count!=1 else ''} detected")
            dur = self.cfg.get("duration_seconds", 0)
            if dur > 0 and elapsed >= dur:
                self._stop_recording()

            # Update trap metrics display and periodic CSV export
            if self.cfg.get("trap_enabled", False):
                fr     = float(self.cfg.get("frame_rate", 30))
                snaps  = [self.trap_stats[i].snapshot(fr)
                          for i in range(min(self.num_cams, 2))]
                labels = ["trap_stats_0", "trap_stats_1"]
                for i, (snap, lbl) in enumerate(zip(snaps, labels)):
                    dpg.set_value(lbl,
                        f"Visits: {snap['visit_count']}  |  "
                        f"Dwell: {snap['mean_dwell_s']:.1f}s  |  "
                        f"Zone ratio: {snap['tz_ratio']:.3f}  |  "
                        f"→: {snap['approach_count']}  ←: {snap['away_count']}")
                if self.num_cams >= 2 and len(snaps) >= 2:
                    ai = snaps[0]["visit_count"] / max(snaps[1]["visit_count"], 1)
                    rd = (snaps[0]["mean_dwell_s"]
                          / max(snaps[1]["mean_dwell_s"], 0.001))
                    dpg.set_value("trap_attractiveness",
                        f"Attractiveness index: {ai:.2f}×  |  "
                        f"Relative dwell ratio: {rd:.2f}×")
                else:
                    dpg.set_value("trap_attractiveness", "(need 2 cameras)")

                # Periodic metrics CSV row
                interval = float(self.cfg.get("metrics_interval_s", 60))
                if (time.time() - self.last_metrics_export) >= interval:
                    self._export_metrics_row()
                    self.last_metrics_export = time.time()

    # ── Mode management ────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self.mode = mode
        if mode == "idle":
            dpg.set_value("mode_text",    "  IDLE")
            dpg.configure_item("mode_text", color=(130,130,150))
            dpg.set_value("elapsed_text", "")
            dpg.set_value("focus_hint",   "")
            dpg.configure_item("record_btn",    label="  RECORD  ",     enabled=True)
            dpg.configure_item("focus_btn",     label="  FOCUS MODE  ", enabled=True)
            dpg.configure_item("reset_cam_btn", enabled=True)
            self._btn_color("record_btn", (180,40,40))
            self._btn_color("focus_btn",  (180,100,20))
            for i in range(2):
                dpg.set_value(f"bug_count_{i}", "--")
            dpg.set_value("trap_stats_0",      "--")
            dpg.set_value("trap_stats_1",      "--")
            dpg.set_value("trap_attractiveness","--")
        elif mode == "recording":
            dpg.set_value("mode_text",  "  RECORDING")
            dpg.configure_item("mode_text", color=(239,83,80))
            dpg.configure_item("record_btn", label="  STOP  ", enabled=True)
            dpg.configure_item("focus_btn",  enabled=False)
            dpg.configure_item("reset_cam_btn", enabled=False)
            self._btn_color("record_btn", (140,20,20))
        elif mode == "focus":
            dpg.set_value("mode_text",  "  FOCUS MODE")
            dpg.configure_item("mode_text", color=(255,167,38))
            dpg.configure_item("focus_btn",     label="  EXIT FOCUS  ", enabled=True)
            dpg.configure_item("record_btn",    enabled=False)
            dpg.configure_item("reset_cam_btn", enabled=False)
            self._btn_color("focus_btn", (160,70,0))
            dpg.set_value("focus_hint",
                "FOCUS MODE - no files saved.  "
                "Adjust lens until the Sharpness bar is maximized.")

    def _toggle_record(self):
        if   self.mode == "idle":      self._start_recording()
        elif self.mode == "recording": self._stop_recording()

    def _toggle_focus(self):
        if   self.mode == "idle":  self._start_focus()
        elif self.mode == "focus": self._stop_focus()

    # ── Recording ─────────────────────────────────────────────────────────────

    def _start_recording(self):
        log(f"_start_recording: {self.num_cams} camera(s), gain={self.cfg.get('gain_db')}")
        if self.num_cams == 0:
            self._alert("No cameras detected."); return

        save_opts = {k: self.cfg.get(k, True)
                     for k in ("save_raw","save_motion","save_csv","save_traj")}
        os.makedirs(self.cfg["save_folder"], exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        names  = [self.cfg["camera_0_name"], self.cfg["camera_1_name"]]
        sf     = self.cfg["save_folder"]

        raw_paths    = [os.path.join(sf,f"{names[i]}_{ts_str}_raw.mp4")    for i in range(self.num_cams)]
        motion_paths = [os.path.join(sf,f"{names[i]}_{ts_str}_motion.mp4") for i in range(self.num_cams)]
        csv_path     =  os.path.join(sf,f"tracks_{ts_str}.csv")
        metrics_path =  os.path.join(sf,f"metrics_{ts_str}.csv")
        self.traj_paths = [os.path.join(sf,f"{names[i]}_{ts_str}_trajmap.png") for i in range(self.num_cams)]

        csv_lock = threading.Lock()
        self.csv_file = None; csv_writer = None
        if save_opts["save_csv"]:
            self.csv_file = open(csv_path,"w",newline="")
            csv_writer    = csv.writer(self.csv_file)
            csv_writer.writerow(["time_s","frame","camera","bug_id",
                                 "cx","cy","area_px2","bbox_x","bbox_y","bbox_w","bbox_h"])

        # Metrics CSV (always written when trap is enabled)
        self.metrics_csv_file   = open(metrics_path, "w", newline="")
        self.metrics_csv_writer = csv.writer(self.metrics_csv_file)
        self.metrics_csv_lock   = threading.Lock()
        self.metrics_csv_writer.writerow([
            "timestamp","elapsed_s",
            "cam0_visits","cam0_mean_dwell_s","cam0_tz_ratio",
            "cam0_approaches","cam0_aways",
            "cam1_visits","cam1_mean_dwell_s","cam1_tz_ratio",
            "cam1_approaches","cam1_aways",
            "attractiveness_index","relative_dwell_ratio",
        ])

        # Reset trap stats
        for _ts in self.trap_stats:
            _ts.reset()

        self.traj_maps      = [TrajectoryMap(self.cfg["image_width"],self.cfg["image_height"]) for _ in range(self.num_cams)]
        self.trackers       = [BugTracker(self.cfg) for _ in range(self.num_cams)]
        self.bug_count_refs = [[0] for _ in range(self.num_cams)]
        self.cfg_refs       = [[dict(self.cfg), {}] for _ in range(self.num_cams)]
        self.raw_writers    = [None]*self.num_cams
        self.motion_writers = [None]*self.num_cams
        self.cameras        = []

        try:
            for i in range(self.num_cams):
                cam = self.cam_list[i]; cam.Init()
                configure_camera(cam, self.cfg, names[i])
                cam.BeginAcquisition(); self.cameras.append(cam)
                if save_opts["save_raw"]:    self.raw_writers[i]    = make_writer(raw_paths[i],    self.cfg)
                if save_opts["save_motion"]: self.motion_writers[i] = make_writer(motion_paths[i], self.cfg)
        except Exception as e:
            self._alert(f"Camera error: {e}"); self._release_cameras(); return

        self._flush_queues()
        self.stop_event = threading.Event(); self.threads = []
        self.record_save_opts = save_opts

        for i in range(self.num_cams):
            t = threading.Thread(
                target=capture_loop,
                args=(self.cameras[i], self.raw_writers[i], self.motion_writers[i],
                      csv_writer, csv_lock, i, names[i],
                      self.stop_event, self.frame_qs[i],
                      self.bug_count_refs[i], self.cfg_refs[i],
                      self.traj_maps[i], self.trackers[i], save_opts,
                      self.trap_stats[i]),
                daemon=True)
            t.start(); self.threads.append(t)

        self.record_t0 = time.time()
        self.last_metrics_export = self.record_t0
        self._set_mode("recording")

    def _stop_recording(self):
        if self.stop_event: self.stop_event.set()
        # Snapshot refs, clear live state immediately
        threads_snap     = list(self.threads)
        cameras_snap     = list(self.cameras)
        raw_snap         = list(self.raw_writers)
        motion_snap      = list(self.motion_writers)
        csv_snap         = self.csv_file
        save_opts_snap   = dict(self.record_save_opts)
        traj_maps_snap   = list(self.traj_maps)
        traj_paths_snap  = list(self.traj_paths)
        save_folder      = self.cfg["save_folder"]
        metrics_file_snap   = self.metrics_csv_file
        metrics_writer_snap = self.metrics_csv_writer
        metrics_lock_snap   = self.metrics_csv_lock
        self.threads     = []; self.cameras = []
        self.raw_writers = [None, None]; self.motion_writers = [None, None]
        self.csv_file    = None
        log("_stop_recording: signalling stop_event")
        # Write final metrics row before clearing the writer
        if metrics_writer_snap:
            self._export_metrics_row(writer=metrics_writer_snap,
                                     lock=metrics_lock_snap)
        self.metrics_csv_file   = None
        self.metrics_csv_writer = None
        # Brief pause to let capture threads finish current img.Release()
        time.sleep(0.06)
        log("_stop_recording: calling EndAcquisition")
        for cam in cameras_snap:
            try: cam.EndAcquisition()
            except Exception as e: log(f"  EndAcquisition error: {e}")
        log("_stop_recording: joining threads")
        for t in threads_snap:
            t.join(timeout=1.0)
        log("_stop_recording: calling DeInit")
        for cam in cameras_snap:
            try: cam.DeInit()
            except Exception as e: log(f"  DeInit error: {e}")
        log("_stop_recording: done")
        self._set_mode("idle")
        # File I/O in background (threads are joined so no more writes happening)
        def _flush():
            for w in raw_snap + motion_snap:
                if w:
                    try: w.release()
                    except Exception: pass
            if csv_snap:
                try: csv_snap.flush(); csv_snap.close()
                except Exception: pass
            if metrics_file_snap:
                try: metrics_file_snap.flush(); metrics_file_snap.close()
                except Exception: pass
            if save_opts_snap.get("save_traj"):
                for tm, tp in zip(traj_maps_snap, traj_paths_snap):
                    if tm:
                        try: cv2.imwrite(tp, tm.canvas)
                        except Exception: pass
            self._pending_alert = (f"Recording saved to:\n{save_folder}", "Done")
        threading.Thread(target=_flush, daemon=True).start()

    # ── Focus mode ─────────────────────────────────────────────────────────────

    def _start_focus(self):
        log(f"_start_focus: {self.num_cams} camera(s), gain={self.cfg.get('gain_db')}")
        if self.num_cams == 0:
            self._alert("No cameras detected."); return
        names = [self.cfg["camera_0_name"], self.cfg["camera_1_name"]]
        self.cameras  = []
        self.cfg_refs = [[dict(self.cfg), {}] for _ in range(self.num_cams)]
        try:
            for i in range(self.num_cams):
                cam = self.cam_list[i]; cam.Init()
                configure_camera(cam, self.cfg, names[i])
                cam.BeginAcquisition(); self.cameras.append(cam)
        except Exception as e:
            self._alert(f"Camera error: {e}"); self._release_cameras(); return
        self._flush_queues()
        self.stop_event = threading.Event(); self.threads = []
        for i in range(self.num_cams):
            t = threading.Thread(
                target=focus_loop,
                args=(self.cameras[i], names[i],
                      self.stop_event, self.frame_qs[i], self.cfg_refs[i]),
                daemon=True)
            t.start(); self.threads.append(t)
        self._set_mode("focus")

    def _stop_focus(self):
        log("_stop_focus: signalling stop_event")
        if self.stop_event: self.stop_event.set()
        threads_snap = list(self.threads)
        cameras_snap = list(self.cameras)
        self.threads = []; self.cameras = []
        # Brief pause — lets focus thread finish current img.Release() before
        # EndAcquisition is called; avoids racing with an outstanding image buffer
        time.sleep(0.06)
        log("_stop_focus: calling EndAcquisition")
        for cam in cameras_snap:
            try: cam.EndAcquisition()
            except Exception as e: log(f"  EndAcquisition error: {e}")
        log("_stop_focus: joining threads")
        for t in threads_snap:
            t.join(timeout=1.0)
        log("_stop_focus: calling DeInit")
        for cam in cameras_snap:
            try: cam.DeInit()
            except Exception as e: log(f"  DeInit error: {e}")
        log("_stop_focus: done")
        self._set_mode("idle")
        for i in range(2):
            dpg.set_value(self.tex_ids[i], blank_texture())

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _save_settings_cb(self):
        try:
            save_settings(self.cfg)
            self._alert(f"Settings saved to:\n{SETTINGS_FILE}", title="Saved")
        except Exception as e:
            self._alert(f"Save error: {e}")

    def _alert(self, msg, title="QQ_Cameras"):
        """Simple modal popup."""
        tag = "alert_modal"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.window(label=title, tag=tag, modal=True,
                        no_resize=True, width=420):
            dpg.add_text(msg, wrap=400)
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=80,
                           callback=lambda: dpg.delete_item(tag))

    def _quit(self):
        if self.mode in ("recording","focus"):
            tag = "confirm_quit"
            if dpg.does_item_exist(tag): dpg.delete_item(tag)
            with dpg.window(label="Quit?", tag=tag, modal=True,
                            no_resize=True, width=340):
                dpg.add_text("Active session running.\nStop and quit?", wrap=320)
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Yes, Quit", width=110,
                                   callback=self._force_quit)
                    dpg.add_spacer(width=10)
                    dpg.add_button(label="Cancel", width=80,
                                   callback=lambda: dpg.delete_item(tag))
        else:
            dpg.stop_dearpygui()

    def _force_quit(self):
        if self.stop_event: self.stop_event.set()
        for t in self.threads: t.join(timeout=2)
        self._release_cameras()
        dpg.stop_dearpygui()

    def _flush_queues(self):
        for q in self.frame_qs:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

    def _release_cameras(self):
        for cam in self.cameras:
            try: cam.EndAcquisition(); cam.DeInit()
            except Exception: pass
        self.cameras        = []
        self.threads        = []
        self.raw_writers    = [None,None]
        self.motion_writers = [None,None]

    def _shutdown(self):
        if self.stop_event: self.stop_event.set()
        for t in self.threads: t.join(timeout=2)
        self._release_cameras()
        if self.csv_file:
            try: self.csv_file.close()
            except Exception: pass
        if self.metrics_csv_file:
            try: self.metrics_csv_file.close()
            except Exception: pass
        if self.cam_list:
            try: self.cam_list.Clear()
            except Exception: pass
        if self.system:
            try: self.system.ReleaseInstance()
            except Exception: pass


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if not HAS_SCIPY:
        print("Note: scipy not found — greedy blob matching used.")
        print("  Better tracking: python3.10 -m pip install scipy")
    app = QQCamerasApp()
    app.run()

if __name__ == "__main__":
    main()
