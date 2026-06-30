"""
Sanity tests for ear_utils.py.

These build FAKE landmark coordinates by hand (no webcam, no face image,
no MediaPipe model needed) to prove the EAR/MAR formulas behave the way
they're supposed to -- open eye shapes score higher than closed ones,
yawns score higher than closed mouths -- before we trust them on live
video. Run directly:

    python test_ear_utils.py
"""

from ear_utils import eye_aspect_ratio, mouth_aspect_ratio, RIGHT_EYE


def make_eye_landmarks(eye_height):
    """Builds a 478-point landmark array where only the right-eye indices
    are meaningfully set. eye_height controls how 'open' the synthetic
    eye shape is (bigger = more open)."""
    landmarks = [(0.0, 0.0)] * 478
    landmarks[33] = (0.0, 0.0)                  # outer corner (p1)
    landmarks[133] = (40.0, 0.0)                 # inner corner (p4), 40px away
    landmarks[160] = (12.0, -eye_height / 2)     # upper lid (p2)
    landmarks[158] = (28.0, -eye_height / 2)     # upper lid (p3)
    landmarks[153] = (28.0, eye_height / 2)      # lower lid (p5)
    landmarks[144] = (12.0, eye_height / 2)      # lower lid (p6)
    return landmarks


def make_mouth_landmarks(mouth_height):
    landmarks = [(0.0, 0.0)] * 478
    landmarks[61] = (0.0, 0.0)                   # left corner
    landmarks[291] = (60.0, 0.0)                  # right corner, 60px away
    landmarks[13] = (30.0, -mouth_height / 2)     # upper inner lip
    landmarks[14] = (30.0, mouth_height / 2)      # lower inner lip
    return landmarks


def test_open_eye_has_higher_ear_than_closed_eye():
    open_eye = make_eye_landmarks(eye_height=14.0)
    closed_eye = make_eye_landmarks(eye_height=2.0)

    ear_open = eye_aspect_ratio(open_eye, RIGHT_EYE)
    ear_closed = eye_aspect_ratio(closed_eye, RIGHT_EYE)

    print(f"EAR (open-eye shape):   {ear_open:.3f}")
    print(f"EAR (closed-eye shape): {ear_closed:.3f}")
    assert ear_open > ear_closed, "open-eye EAR should exceed closed-eye EAR"
    assert ear_closed < 0.15, "closed-eye EAR should land in the 'drowsy' range"
    assert ear_open > 0.20, "open-eye EAR should land in the 'awake' range"
    print("PASS: EAR correctly distinguishes open vs closed eye shapes\n")


def test_fully_shut_eye_gives_zero_ear():
    shut = make_eye_landmarks(eye_height=0.0)
    ear_shut = eye_aspect_ratio(shut, RIGHT_EYE)
    print(f"EAR (fully shut, height=0): {ear_shut:.3f}")
    assert ear_shut == 0.0
    print("PASS: fully shut eye gives EAR = 0\n")


def test_yawn_raises_mar_above_closed_mouth():
    closed_mouth = make_mouth_landmarks(mouth_height=2.0)
    yawning = make_mouth_landmarks(mouth_height=50.0)

    mar_closed = mouth_aspect_ratio(closed_mouth)
    mar_yawn = mouth_aspect_ratio(yawning)

    print(f"MAR (mouth closed): {mar_closed:.3f}")
    print(f"MAR (yawning):      {mar_yawn:.3f}")
    assert mar_yawn > mar_closed, "yawning should raise MAR above closed-mouth MAR"
    assert mar_yawn > 0.55, "wide yawn should clear a typical 0.55 threshold"
    assert mar_closed < 0.55, "closed mouth should stay under a typical 0.55 threshold"
    print("PASS: MAR correctly rises during a yawn\n")


if __name__ == "__main__":
    test_open_eye_has_higher_ear_than_closed_eye()
    test_fully_shut_eye_gives_zero_ear()
    test_yawn_raises_mar_above_closed_mouth()
    print("All EAR/MAR sanity tests passed.")
