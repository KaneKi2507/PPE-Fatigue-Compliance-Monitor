"""
Pure geometry functions for drowsiness detection signals:
  - Eye Aspect Ratio (EAR): drops when eyes close
  - Mouth Aspect Ratio (MAR): rises when yawning

These operate on a list of (x, y) landmark coordinates only -- no image
or camera involved -- which is exactly what makes them unit-testable in
isolation (see test_ear_utils.py) before we ever point a webcam at them.

Landmark indices below refer to MediaPipe's 478-point face mesh topology,
used by both the legacy FaceMesh solution and the current FaceLandmarker
Tasks API (the indices didn't change between the two -- only the API
that produces them did). The 6-point-per-eye convention follows
Soukupova & Cech, "Real-Time Eye Blink Detection using Facial Landmarks"
(2016), adapted to MediaPipe's mesh point IDs.
"""

import math

# Each eye: [outer_corner, upper_1, upper_2, inner_corner, lower_1, lower_2]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]    # subject's right eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]    # subject's left eye

# Mouth: corners + inner upper/lower lip center, for a simple open/closed ratio
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_UPPER_INNER = 13
MOUTH_LOWER_INNER = 14


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_indices):
    """
    landmarks: indexable sequence of (x, y) points, one per face-mesh point
               (478 of them if you pass the full set straight from MediaPipe)
    eye_indices: RIGHT_EYE or LEFT_EYE above

    Returns a float. Roughly 0.25-0.35 for a comfortably open eye, falling
    toward 0.05-0.15 as it closes. Exact numbers depend on your face and
    camera angle -- always calibrate the threshold against your own face
    (see calibrate_thresholds.py) rather than trusting a textbook number.
    """
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_indices)
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def average_ear(landmarks):
    """Average of both eyes. More stable than either eye alone -- it
    smooths out slight head turns or one eye being briefly occluded."""
    left = eye_aspect_ratio(landmarks, LEFT_EYE)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    return (left + right) / 2.0


def mouth_aspect_ratio(landmarks):
    """Same idea as EAR but for the mouth -- rises sharply during a yawn."""
    top = landmarks[MOUTH_UPPER_INNER]
    bottom = landmarks[MOUTH_LOWER_INNER]
    left = landmarks[MOUTH_LEFT_CORNER]
    right = landmarks[MOUTH_RIGHT_CORNER]
    vertical = _dist(top, bottom)
    horizontal = _dist(left, right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal
