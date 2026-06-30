## Setup (Main thing)

**1. Open a terminal (cmd), preferrably with Run as Administrator** Change Directory to where the project folder is cloned/downloaded.
for eg: cd C:\Users\AYUSH\Downloads\ppe-fatigue-monitor (1)\ppe-fatigue-monitor.

**2. After cd C:\.... install the requirements:**
```bash
pip install -r requirements.txt
```

**3. Run these tests first** (proves the EAR/MAR math is correct,
takes a few seconds, no camera needed):
```bash
cd src
python test_ear_utils.py
```
All three tests should print PASS.

**4. Run the live fatigue monitor:**
```bash
python fatigue_detection.py
```
First run downloads a small (~4MB) face landmark model automatically.
A window opens showing your webcam feed with live EAR/MAR numbers and an
AWAKE/DROWSY status. Press `q` to quit.

### Ignore this shit below (keeping it cuz idk kab kaam aajaaye):

# PPE + Fatigue Compliance Monitor

InnoVent-27 submission -- Industrial Heavy Machinery / Operator Safety &
Human-Machine Interaction. Detects missing PPE (hard hat / safety vest)
and operator drowsiness in real time from a single camera feed, designed
to run fully offline at the edge.

## Project status

| Step | What | Status |
|---|---|---|
| 1 | Fatigue/drowsiness detection (Mediapipe + EAR/MAR) | **Built + tested** |
| 2 | PPE detection model training (Colab notebook) | **Notebook ready -- run on Colab** |
| 3 | PPE local inference script | Not started |
| 4 | Fusion + alert layer | Not started |
| 5 | Streamlit dashboard + logging | Not started |
| 6 | Real-world testing, demo script, edge-deployment writeup | Not started |

## Project structure

```
ppe-fatigue-monitor/
├── requirements.txt
├── src/
│   ├── ear_utils.py          # pure EAR/MAR math, no camera needed
│   ├── test_ear_utils.py     # sanity tests for the above (run these first)
│   └── fatigue_detection.py  # live webcam drowsiness monitor
├── notebooks/
│   └── train_ppe_yolo.ipynb  # run this on Google Colab (needs GPU)
├── models/                   # put best.pt here after training on Colab
└── logs/                     # fatigue_log.csv gets written here at runtime
```



**Calibrate before your demo** -- the default thresholds
(`EAR_DROWSY_THRESHOLD = 0.21`, `MAR_YAWN_THRESHOLD = 0.55`) are
reasonable starting points, not guaranteed-correct for every face, camera,
and lighting setup:
1. Watch the on-screen EAR/MAR numbers while blinking and talking normally.
2. Then deliberately close your eyes for 2s and yawn once, and note those
   numbers.
3. Edit the two threshold constants near the top of `fatigue_detection.py`
   to sit between your "normal" and "drowsy/yawning" ranges.

**3. Train the PPE model:** open `notebooks/train_ppe_yolo.ipynb` in
Google Colab (`Runtime > Change runtime type > T4 GPU` first), follow the
cells in order, and download `best.pt` into `models/` when done. Full
instructions are inside the notebook.

## Gotchas found while building this (worth knowing, not just trivia)

- **Mediapipe's old `mp.solutions.face_mesh.FaceMesh` API is gone** in
  current mediapipe versions (0.10.x+) -- it raises
  `AttributeError: module 'mediapipe' has no attribute 'solutions'`. A lot
  of tutorials/AI-generated snippets still use it. `fatigue_detection.py`
  uses the current `FaceLandmarker` Tasks API instead -- the underlying
  478-point landmark layout is unchanged, so `ear_utils.py` didn't need to
  care about the difference.
- **`opencv-python-headless` cannot open a GUI window.** If `cv2.imshow()`
  fails with an error about Qt/GTK or "not implemented," you've got the
  headless package installed instead of the full `opencv-python`. Check
  with `pip show opencv-python-headless` -- if it's present, `pip uninstall`
  it and make sure plain `opencv-python` is installed instead.
- **The audio beep needs the PortAudio system library**, which isn't
  installed on every machine. `play_beep()` in `fatigue_detection.py`
  already handles this -- it tries the real beep and falls back to a
  terminal bell (`\a`) without crashing if PortAudio (or any audio device)
  isn't available. You don't need to fix anything for the script to keep
  working; this only matters if you actually want the beep sound itself.

## Recommended dataset for Step 2

[Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
on Roboflow Universe -- already in YOLOv8 format with `Hardhat`,
`NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, and `Person` classes
(plus a few extras). A no-signup Kaggle mirror is linked inside the
notebook as a fallback.
