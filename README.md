# PPE + Fatigue Compliance Monitor
### Tata Technologies InnoVent-27 | Industrial Heavy Machinery | Operator Safety & Human-Machine Interaction

---

## What We Are Building

A real-time safety monitoring system for construction and heavy machinery sites that uses a single camera to detect two things simultaneously:

1. **Is the worker wearing the required PPE?** — hard hat and safety vest
2. **Is the worker alert and awake?** — not drowsy, not falling asleep on the job

When either condition fails, the system raises an instant visual and audio alert and logs the violation with a timestamp.

The system is designed to run **fully offline**, with no internet connection required during operation — because most real construction and mining sites in India have poor or zero network connectivity. This is what is meant by "Edge AI" — the AI runs locally on the device itself, not on a cloud server somewhere.

---

## Build Plan — Step by Step

### Step 1 — Fatigue / Drowsiness Detector ✅ DONE (Setup below after mentioning all the steps)
**What it does:** Opens the webcam, tracks facial landmarks using MediaPipe, computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), and triggers a DROWSY alert if eyes stay closed for ~0.5 seconds or a yawn is detected.

**Files:**
- `src/ear_utils.py` — The core EAR/MAR math
- `src/test_ear_utils.py` — Tests to verify the math is correct (run this first)
- `src/fatigue_detection.py` — The live webcam script

**Status:** Built, tested, and confirmed working.

---

### Step 2 — Train the PPE Detection Model (Google Colab)
**What it does:** Uses a publicly available, pre-labeled dataset of construction site images (thousands of photos with hard hat / no hard hat / vest / no vest labels already marked) to train a YOLOv8 nano model. The training runs on a free Colab GPU and takes roughly 30-60 minutes unattended.

The output is a file called `best.pt` — this is the trained model's "brain." It gets downloaded to your laptop and placed in the `models/` folder.

**Dataset used:** [Construction Site Safety — Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
Classes: `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Person`

**Files:**
- `notebooks/train_ppe_yolo.ipynb` — Open this in Google Colab and follow the instructions inside

**Status:** Notebook ready. Needs to be run on Colab to produce `best.pt`.

---

### Step 3 — PPE Local Inference Script
**What it does:** Takes the trained `best.pt` model and runs it live on your webcam. Draws bounding boxes around detected people and labels them with their PPE status — helmet on/off, vest on/off — in real time on your laptop, no internet needed.

**Files to be created:**
- `src/ppe_detection.py`

**Status:** Not started. Will be built after Step 2 is complete and `best.pt` is available.

---

### Step 4 — Fusion: Combine Both Detectors
**What it does:** Merges the two separate scripts (fatigue detector + PPE detector) into a single program that processes each camera frame through both models at the same time and produces a unified alert status. A single loop, a single window, both detections running in parallel.

**Files to be created:**
- `src/monitor.py` — The combined detection loop

**Status:** Not started. Depends on Steps 1 and 3 being complete.

---

### Step 5 — Alert System + Violation Logging
**What it does:** Every time a violation is detected (missing PPE or drowsiness), the system:
- Flashes a red border on the screen
- Plays an audio beep
- Writes a row to `logs/violations.csv` with: timestamp, worker ID (based on position in frame), violation type, and the confidence score of the detection

This log file is your evidence that the system works over time — it can be shown to judges as a real output, not just a live demo.

**Status:** Partially done (beep + basic CSV logging already exist in the fatigue detector). Needs to be expanded for the combined system in Step 5.

---

### Step 6 — Streamlit Dashboard
**What it does:** Wraps the entire system in a clean visual interface. Instead of just a raw camera window with text overlaid, this shows a proper dashboard with the live feed on one side and a running violation log on the other — the kind of thing that looks like a real deployable product.

**Files to be created:**
- `dashboard/app.py`

**Status:** Not started. Built last, once the detection logic is stable.

---

### Step 7 — Edge Deployment Story (Benchmark + Optional Pi Test)
**What it does:** Exports the trained model to ONNX format (a lightweight format that runs without needing PyTorch installed) and benchmarks how fast it runs using only CPU — no GPU. This produces real numbers we can show judges as proof the system is viable on cheap edge hardware like a Raspberry Pi.

**Status:** Not started. Done after Step 6.

---


## Setup for Stage 1:

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

**Common issues:**
- *Camera window doesn't open / black screen:* Make sure you installed `opencv-python` and NOT `opencv-python-headless` (check with `pip show opencv-python`).
- *"Allow camera access" popup on Mac:* Click Allow. If you accidentally clicked Deny, go to System Settings → Privacy → Camera and re-enable it for your terminal app.
- *No beep sound:* Not a problem — the script beeps if it can, but falls back to a terminal bell silently if no audio device is available. Everything else still works.

---

## Folder Structure

```
ppe-fatigue-monitor/
├── README.md                    ← You are here
├── requirements.txt             ← All Python libraries needed
├── src/
│   ├── ear_utils.py             ← EAR/MAR math (Step 1)
│   ├── test_ear_utils.py        ← Tests for the above (Step 1)
│   ├── fatigue_detection.py     ← Live drowsiness detector (Step 1)
│   ├── ppe_detection.py         ← PPE detector (Step 3, coming soon)
│   └── monitor.py               ← Combined system (Step 4, coming soon)
├── notebooks/
│   └── train_ppe_yolo.ipynb     ← Colab training notebook (Step 2)
├── dashboard/
│   └── app.py                   ← Streamlit dashboard (Step 6, coming soon)
├── models/
│   └── best.pt                  ← Put your trained weights here after Step 2
└── logs/
    └── violations.csv           ← Auto-generated at runtime, not committed to git
```

---