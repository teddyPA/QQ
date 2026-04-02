# QQ_Cameras — Feature Roadmap

## Completed

### ✅ Wire Exclusion (Test Mode)
When testing with a physical bug on a wire, the wire itself triggered motion detection as a false positive.
- Added `wire_aspect_ratio` setting (default 4.0) in the settings panel
- Contours with `max(w,h) / min(w,h) > wire_aspect_ratio` are discarded before tracking
- Set to 0 to disable

### ✅ Trap Attractiveness Metrics (Dual-Camera Experiment)
**Setup:**
- Camera 0 = treatment trap (with attractants)
- Camera 1 = control trap (no attractants)

**Implemented metrics:**
- **Visit count** — unique bug track IDs that entered the trap zone ROI
- **Dwell time** — mean time each tracked bug spends inside the zone (frames → seconds)
- **Approach/away counts** — frames a bug moved toward vs. away from the trap centre
- **Trap-zone entry ratio** — visits / total detections (normalises for overall activity)
- **Attractiveness index** — Camera 0 visit count / Camera 1 visit count (>1 = treatment more attractive)
- **Relative dwell ratio** — Camera 0 mean dwell / Camera 1 mean dwell

**UI additions:**
- Per-camera ROI: set Centre X, Centre Y, Radius in the "Trap Attractiveness Metrics" settings section
- The trap zone circle is overlaid live on the camera preview
- Live stats panel below the camera previews showing all metrics in real time
- Metrics CSV exported at session end **and** as a new row every N seconds (configurable via "Export interval" setting, default 60 s)

### ✅ Camera Reset
- Cameras are automatically reset (Init → EndAcquisition → DeInit cycle) every time the program starts, clearing any stale state from a previous crash
- **Reset Cameras** button in the UI — reinitialises and re-enumerates cameras without needing to unplug/replug

---

## Pending Enhancements

<!-- Add new feature requests below -->

