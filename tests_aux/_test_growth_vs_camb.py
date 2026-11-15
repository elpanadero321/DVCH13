"""Prueba independiente: el solver de crecimiento de alpha_general_fit
(Python, 2 fluidos, gauge sincrono) vs CAMB 2.0.4 (codigo independiente)
para LCDM. Si coinciden, la maquinaria de crecimiento esta validada
externamente. f*sigma8(z) en los 6 puntos RSD, ambos normalizados a
sigma8(0) = 0.81."""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/DVCH13-1")
sys.path.insert(0, os.environ.get("DVCH_CAMB_ROOT",
                                   "/mnt/d/DVCH-external/CAMB-master"))
import camb  # noqa: F401
from camb import get_results
from camb.model import CAMBparams

import alpha_general_fit as ag

RSD_Z = np.array([0.02, 0.15, 0.32, 0.57, 0.77, 1.23])
SIG8_REF = 0.81
OM = 0.315
Z_OUT = np.concatenate([[0.0], RSD_Z])   # z=0 incluido para sigma8(0)

p = CAMBparams()
p.set_cosmology(H0=67.4, ombh2=0.0223,
                omch2=(OM - 0.0223 / 0.674**2) * 0.674**2,
                omk=0.0, tau=0.054, mnu=0.0, neutrino_hierarchy="normal")
p.set_matter_power(redshifts=list(Z_OUT), kmax=0.3, silent=True)
p.InitPower.set_params(As=2.1e-9, ns=0.965)
res = get_results(p)
fs8_all = np.array(res.get_fsigma8(), dtype=float)
# CAMB devuelve fs8 en redshifts ORDENADOS DE MAYOR A MENOR: invertir y
# quitar el z=0 (primera columna de Z_OUT es 0, que cae al final).
fs8_camb = fs8_all[::-1][1:]
s8_camb0 = float(res.get_sigma8_0())
fs8_camb = fs8_camb * (SIG8_REF / s8_camb0)
print(f"CAMB: sigma8(0) = {s8_camb0:.4f} (renormalizado a {SIG8_REF})")

Nn = np.log(ag.A_START)
_, Dc, _ = ag.growth_two_fluid(OM, 0.0, 0.0)
fs8_py = []
for z in RSD_Z:
    N = np.log(1.0 / (1.0 + z))
    h = 1e-4
    f = (Dc(N + h) - Dc(N - h)) / (2 * h * Dc(N))
    fs8_py.append(f * SIG8_REF * Dc(N) / Dc(0.0))
fs8_py = np.array(fs8_py)

print(f"\n{'z':>6} {'CAMB':>8} {'Python':>8} {'diff':>9}")
for z, c, m in zip(RSD_Z, fs8_camb, fs8_py):
    print(f"{z:6.2f} {c:8.4f} {m:8.4f} {(m/c-1)*100:+8.2f}%")
print(f"\nmax |diff|: {np.max(np.abs(fs8_py/fs8_camb-1))*100:.2f}%")
