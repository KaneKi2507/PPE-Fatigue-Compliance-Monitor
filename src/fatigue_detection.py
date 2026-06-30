"""
Fatigue / drowsiness detector -- webcam based, runs on CPU.

Pipeline:
  webcam frame -> MediaPipe FaceLandmarker -> 478 face landmarks
               -> EAR (eye closure) + MAR (yawn) via ear_utils.py
               -> consecutive-frame debounce -> AWAKE / DROWSY state
               -> on-screen alert + beep + CSV log line

Run:
    python fatigue_detection.py

Quit:
    press 'q' with the camera window focused

----------------------------------------------------------------------
IMPORTANT -- MediaPipe API note:

This uses the CURRENT FaceLandmarker (Tasks API). A lot of older
tutorials use `mp.solutions.face_mesh.FaceMesh(...)`, which has been
REMOVED as of mediapipe>=0.10.x -- that pattern now raises
`AttributeError: module 'mediapipe' has no attribute 'solutions'`.
The underlying 478-point landmark layout is identical between the old
and new API, so the EAR/MAR math in ear_utils.py is unaffected -- only
the setup code changed. Verified against mediapipe==0.10.33.
----------------------------------------------------------------------

CALIBRATION -- do this before your demo, on your own face/lighting:
  1. Run the script. The top-right corner shows live EAR / MAR numbers.
  2. Blink and talk normally for ~10s -- note the EAR/MAR range.
  3. Deliberately close your eyes for 2s and yawn once -- note those numbers.
  4. Set EAR_DROWSY_THRESHOLD a little above your "closed" EAR and
     MAR_YAWN_THRESHOLD a little below your "yawning" MAR.
  Defaults below are reasonable starting points, not guaranteed-correct
  for every face/camera/lighting combination.
"""

import csv
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from ear_utils import average_ear, mouth_aspect_ratio

# ---------------------------------------------------------------- config --
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "..", "models", "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

EAR_DROWSY_THRESHOLD = 0.21       # below this -> eye counts as "closing"
MAR_YAWN_THRESHOLD = 0.55         # above this -> mouth counts as "yawning"
CONSEC_FRAMES_FOR_ALERT = 15      # ~0.5s at 30fps -- avoids alerting on a normal blink
ALERT_COOLDOWN_SECONDS = 4.0      # don't re-beep every single frame once triggered

LOG_PATH = os.path.join(THIS_DIR, "..", "logs", "fatigue_log.csv")
# ----------------------------------------------------------------------- --


def ensure_model_downloaded():
    """Downloads the FaceLandmarker model file once, on first run.
    Needs normal internet access -- this step can't be pre-run in a
    sandboxed/offline environment, so do this on the machine you'll
    actually demo from, ahead of time."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmark model (~4MB) to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def play_beep():
    """Best-effort audio alert via sounddevice (already a mediapipe
    dependency, so no extra install needed). Falls back to a terminal
    bell and never crashes the main loop if no audio device exists --
    some lab/demo machines won't have one wired up."""
    try:
        import sounddevice as sd
        duration_s, freq, sr = 0.25, 1000, 44100
        t = np.linspace(0, duration_s, int(sr * duration_s), False)
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        sd.play(tone, sr)
    except Exception:
        print("\a", end="", flush=True)


def init_logger():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    f = open(LOG_PATH, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["timestamp", "event", "ear", "mar"])
        f.flush()
    return f, writer


def main():
    ensure_model_downloaded()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,  # not used in the alert logic yet, but
                                        # gives eyeBlinkLeft/Right confidence
                                        # scores as a free second signal if
                                        # you want to cross-check EAR later
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam at index 0. If you have multiple cameras, "
            "try cv2.VideoCapture(1); also check OS camera permissions."
        )

    log_file, log_writer = init_logger()

    drowsy_frame_count = 0
    last_alert_time = 0.0
    start_time = time.time()

    print("Fatigue monitor running. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, stopping.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            status_text, status_color = "NO FACE", (128, 128, 128)
            ear_value = mar_value = 0.0

            if result.face_landmarks:
                h, w = frame.shape[:2]
                points = [(p.x * w, p.y * h) for p in result.face_landmarks[0]]

                ear_value = average_ear(points)
                mar_value = mouth_aspect_ratio(points)

                drowsy_now = (ear_value < EAR_DROWSY_THRESHOLD) or (mar_value > MAR_YAWN_THRESHOLD)
                drowsy_frame_count = drowsy_frame_count + 1 if drowsy_now else 0

                if drowsy_frame_count >= CONSEC_FRAMES_FOR_ALERT:
                    status_text, status_color = "DROWSY", (0, 0, 255)
                    now = time.time()
                    if now - last_alert_time > ALERT_COOLDOWN_SECONDS:
                        play_beep()
                        log_writer.writerow(
                            [datetime.now().isoformat(), "DROWSY_ALERT",
                             f"{ear_value:.3f}", f"{mar_value:.3f}"]
                        )
                        log_file.flush()
                        last_alert_time = now
                else:
                    status_text, status_color = "AWAKE", (0, 200, 0)

            # --- overlay ---
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
            cv2.putText(frame, f"Status: {status_text}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            cv2.putText(frame, f"EAR: {ear_value:.2f}  MAR: {mar_value:.2f}",
                        (330, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if status_text == "DROWSY":
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                              (0, 0, 255), 8)

            cv2.imshow("Fatigue Monitor (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        landmarker.close()


if __name__ == "__main__":
    main()
