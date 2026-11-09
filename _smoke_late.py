import os
import sys

sys.path.insert(0, "/mnt/d/DVCH13-1")
os.chdir("/mnt/d/DVCH13-1")
import numpy as np

from run_alpha_joint import chi2_late_from_camb

for alpha in (0.0, 0.15, 0.6):
    try:
        c2 = chi2_late_from_camb(67.36, 0.02237, 0.1200, alpha, -19.4)
        print(f"alpha={alpha}: chi2_late = {c2:.2f}")
    except Exception as exc:
        print(f"alpha={alpha}: ERROR {type(exc).__name__}: {exc}")
