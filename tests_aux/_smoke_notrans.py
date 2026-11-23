"""Smoke test: una evaluacion de la likelihood conjunta del no-transferir."""
import os
import sys

sys.path.insert(0, "/mnt/d/DVCH13-1")
os.chdir("/mnt/d/DVCH13-1")

import run_notrans_joint as rn

# punto cercano al mejor ajuste tardio (Om=0.319, n=0.249, beta=0.665)
lp = rn._joint_loglike(
    H0=68.0, ombh2=0.0224, omch2=0.1126, tau=0.054, ln10As=3.04, ns=0.965,
    n=0.249, beta=0.665, Msn=-19.408)
print(f"loglike conjunta (no-transferir, punto de prueba) = {lp:.3f}")
print(f"-> chi2 total = {-2.0 * lp:.2f}   (LCDM control: 4178.59)")
