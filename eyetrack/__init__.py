"""
eyetrack - webcam eye-tracking pipeline for capturing real-time student
behaviour (SIGCSE 2026 poster pipeline).

Stages: face/eye detection (MediaPipe) -> gaze (geometric or MPIIGaze/EVE
CNNs) -> eye events (I-DT fixations, saccades, blinks) -> windowed features
-> behaviour classification -> educator alerts and researcher logs.

Run ``python -m eyetrack -h`` for the command-line interface.
"""

__version__ = "3.0.0"
