"""
QQ_Cameras — FLIR Mosquito Motion Detection & Tracking
=======================================================
Version   : 2.2
Build date: 2026-04-01

GUI: Dear PyGui (GPU-rendered, no Tcl/Tk dependency)
  - Live camera preview per camera
  - Scrollable settings panel with min/max for every field
  - RECORD button (toggle) — saves to ~/Downloads/QQ
  - FOCUS MODE button — sharpness bar + crosshair, no files saved
  - Per-file output toggles (raw, motion, CSV, trajectory map)

Install:
    python3.10 -m pip install "numpy<2" opencv-python scipy dearpygui
    PySpin: cd ~/Downloads
            tar -xzf /Applications/Spinnaker/PySpin/spinnaker_python-4.3.0.189-cp310-cp310-macosx_13_0_arm64.tar.gz
            python3.10 -m pip install spinnaker_python-4.3.0.189-cp310-cp310-macosx_13_0_arm64.whl
"""

VERSION    = "2.2"
BUILD_DATE = "2026-04-01"

import os, json, threading, time, sys, csv, queue
from datetime import datetime
from collections import deque

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
}
SETTINGS_FILE = "camera_settings.json"

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
}

TRAIL_COLORS = [
    (0,255,0),(0,200,255),(255,100,0),(180,0,255),
    (255,255,0),(0,80,255),(255,0,180),(0,255,150),
    (255,50,50),(150,255,50),(50,150,255),(255,200,50),
]

# Preview texture dimensions
PREV_W, PREV_H = 440, 310

# ──────────────────────────────────────────────────────────────────────────────
#  SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
def settings_path(cfg):
    return os.path.join(cfg["save_folder"], SETTINGS_FILE)

def load_settings():
    cfg  = dict(DEFAULTS)
    path = os.path.join(DEFAULTS["save_folder"], SETTINGS_FILE)
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def save_settings(cfg):
    os.makedirs(cfg["save_folder"], exist_ok=True)
    with open(settings_path(cfg), "w") as f:
        json.dump(cfg, f, indent=2)

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
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= cfg["min_bug_area"]:
            x,y,w,h = cv2.boundingRect(cnt)
            dets.append((x+w/2, y+h/2, area, x, y, w, h))
    return cleaned, dets

def render_motion_frame(mono, tracked, tracker):
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
                 bug_count_ref, cfg_ref, traj_map, tracker, save_opts):
    bg_sub      = make_bg_subtractor()
    frame_count = 0
    t0          = time.time()
    while not stop_event.is_set():
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
            motion_frame = render_motion_frame(mono, tracked, tracker)
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
    """Convert a BGR OpenCV frame to a flat RGBA float list for Dear PyGui."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (PREV_W, PREV_H))
    rgba    = np.ones((PREV_H, PREV_W, 4), dtype=np.float32)
    rgba[:,:,:3] = resized.astype(np.float32) / 255.0
    return rgba.flatten().tolist()

def blank_texture():
    """Dark placeholder texture."""
    data = np.full((PREV_H, PREV_W, 4), 0.06, dtype=np.float32)
    data[:,:,3] = 1.0
    return data.flatten().tolist()

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

        # Runtime
        self.cameras        = []
        self.stop_event     = threading.Event()
        self.threads        = []
        self.frame_qs       = [queue.Queue(maxsize=3), queue.Queue(maxsize=3)]
        self.bug_count_refs = [[0], [0]]
        self.cfg_refs       = [[dict(self.cfg)], [dict(self.cfg)]]
        self.traj_maps      = [None, None]
        self.trackers       = [None, None]
        self.raw_writers    = [None, None]
        self.motion_writers = [None, None]
        self.csv_file       = None
        self.record_t0      = None
        self.traj_paths     = []
        self.record_save_opts = {}

    # ── Dear PyGui setup ───────────────────────────────────────────────────────

    def run(self):
        dpg.create_context()
        self._apply_theme()

        # Texture registry — one texture per camera slot
        with dpg.texture_registry():
            self.tex_ids = [
                dpg.add_dynamic_texture(PREV_W, PREV_H, blank_texture()),
                dpg.add_dynamic_texture(PREV_W, PREV_H, blank_texture()),
            ]

        self._build_ui()

        dpg.create_viewport(
            title=f"QQ_Cameras  v{VERSION}  |  {BUILD_DATE}",
            width=1220, height=780,
            min_width=900, min_height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_win", True)

        # Main render loop
        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()

        self._shutdown()
        dpg.destroy_context()

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

                # ── Right: previews + status + buttons ────────────────────────
                with dpg.group():

                    # Camera previews
                    with dpg.group(horizontal=True):
                        for i in range(2):
                            with dpg.group():
                                dpg.add_text(names[i].upper(),
                                             color=(79,195,247))
                                dpg.add_image(self.tex_ids[i],
                                              width=PREV_W, height=PREV_H)
                                dpg.add_text("--",
                                             tag=f"bug_count_{i}",
                                             color=(100,100,130))
                            if i == 0:
                                dpg.add_spacer(width=12)

                    dpg.add_spacer(height=6)
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
        self._int_row("min_bug_area",       "Min bug area")
        self._int_row("blob_dilation",      "Blob dilation")

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

    # ── Config helpers ─────────────────────────────────────────────────────────

    def _cfg_set(self, key, value):
        self.cfg[key] = value
        if key == "save_folder":
            dpg.set_value("folder_label", f"Output -> {value}")
        for cr in self.cfg_refs:
            cr[0] = dict(self.cfg)

    # ── Per-tick update ────────────────────────────────────────────────────────

    def _tick(self):
        # Push camera frames to textures
        for i in range(min(self.num_cams, 2)):
            try:
                frame = self.frame_qs[i].get_nowait()
                dpg.set_value(self.tex_ids[i], frame_to_texture(frame))
            except queue.Empty:
                pass

        # Update status
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

    # ── Mode management ────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self.mode = mode
        if mode == "idle":
            dpg.set_value("mode_text",    "  IDLE")
            dpg.configure_item("mode_text", color=(130,130,150))
            dpg.set_value("elapsed_text", "")
            dpg.set_value("focus_hint",   "")
            dpg.configure_item("record_btn", label="  RECORD  ", enabled=True)
            dpg.configure_item("focus_btn",  label="  FOCUS MODE  ", enabled=True)
            self._btn_color("record_btn", (180,40,40))
            self._btn_color("focus_btn",  (180,100,20))
            for i in range(2):
                dpg.set_value(f"bug_count_{i}", "--")
        elif mode == "recording":
            dpg.set_value("mode_text",  "  RECORDING")
            dpg.configure_item("mode_text", color=(239,83,80))
            dpg.configure_item("record_btn", label="  STOP  ", enabled=True)
            dpg.configure_item("focus_btn",  enabled=False)
            self._btn_color("record_btn", (140,20,20))
        elif mode == "focus":
            dpg.set_value("mode_text",  "  FOCUS MODE")
            dpg.configure_item("mode_text", color=(255,167,38))
            dpg.configure_item("focus_btn",  label="  EXIT FOCUS  ", enabled=True)
            dpg.configure_item("record_btn", enabled=False)
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
        if self.num_cams == 0:
            self._alert("No cameras detected."); return

        save_opts = {k: self.cfg.get(k, True)
                     for k in ("save_raw","save_motion","save_csv","save_traj")}
        os.makedirs(self.cfg["save_folder"], exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        names = [self.cfg["camera_0_name"], self.cfg["camera_1_name"]]
        sf    = self.cfg["save_folder"]

        raw_paths    = [os.path.join(sf,f"{names[i]}_{ts}_raw.mp4")    for i in range(self.num_cams)]
        motion_paths = [os.path.join(sf,f"{names[i]}_{ts}_motion.mp4") for i in range(self.num_cams)]
        csv_path     =  os.path.join(sf,f"tracks_{ts}.csv")
        self.traj_paths = [os.path.join(sf,f"{names[i]}_{ts}_trajmap.png") for i in range(self.num_cams)]

        csv_lock = threading.Lock()
        self.csv_file = None; csv_writer = None
        if save_opts["save_csv"]:
            self.csv_file = open(csv_path,"w",newline="")
            csv_writer    = csv.writer(self.csv_file)
            csv_writer.writerow(["time_s","frame","camera","bug_id",
                                 "cx","cy","area_px2","bbox_x","bbox_y","bbox_w","bbox_h"])

        self.traj_maps      = [TrajectoryMap(self.cfg["image_width"],self.cfg["image_height"]) for _ in range(self.num_cams)]
        self.trackers       = [BugTracker(self.cfg) for _ in range(self.num_cams)]
        self.bug_count_refs = [[0] for _ in range(self.num_cams)]
        self.cfg_refs       = [[dict(self.cfg)] for _ in range(self.num_cams)]
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
                      self.traj_maps[i], self.trackers[i], save_opts),
                daemon=True)
            t.start(); self.threads.append(t)

        self.record_t0 = time.time()
        self._set_mode("recording")

    def _stop_recording(self):
        if self.stop_event: self.stop_event.set()
        for t in self.threads: t.join(timeout=5)
        for w in self.raw_writers + self.motion_writers:
            if w:
                try: w.release()
                except Exception: pass
        if self.csv_file:
            try: self.csv_file.flush(); self.csv_file.close()
            except Exception: pass
            self.csv_file = None
        if self.record_save_opts.get("save_traj"):
            for i in range(self.num_cams):
                if self.traj_maps[i]:
                    try: cv2.imwrite(self.traj_paths[i], self.traj_maps[i].canvas)
                    except Exception: pass
        self._release_cameras()
        self._set_mode("idle")
        self._alert(f"Recording saved to:\n{self.cfg['save_folder']}", title="Done")

    # ── Focus mode ─────────────────────────────────────────────────────────────

    def _start_focus(self):
        if self.num_cams == 0:
            self._alert("No cameras detected."); return
        names = [self.cfg["camera_0_name"], self.cfg["camera_1_name"]]
        self.cameras  = []
        self.cfg_refs = [[dict(self.cfg)] for _ in range(self.num_cams)]
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
        if self.stop_event: self.stop_event.set()
        for t in self.threads: t.join(timeout=3)
        self._release_cameras()
        self._set_mode("idle")
        for i in range(2):
            dpg.set_value(self.tex_ids[i], blank_texture())

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _save_settings_cb(self):
        try:
            save_settings(self.cfg)
            self._alert(f"Settings saved to:\n{settings_path(self.cfg)}", title="Saved")
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
