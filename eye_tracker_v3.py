"""
Eye Tracker V3 - behaviour-classification pipeline.

Thin wrapper so the newest version can be started like V1/V2::

    python eye_tracker_v3.py live --calibrate --aoi configs/aoi_programming.yaml --probes

Equivalent to ``python -m eyetrack ...``; see README_V3.md.
"""

from eyetrack.app import main

if __name__ == "__main__":
    main()
