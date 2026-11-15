import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, "/mnt/d/DVCH13-1")
import dvch_vs_lcdm_real as dvr

OR0 = dvr.Omega_r0
RSD_Z = np.array([0.02, 0.15, 0.32, 0.57, 0.77, 1.23])
RSD_FS8 = np.array([0.428, 0.45, 0.44, 0.42, 0.39, 0.35])
RSD_ERR = np.array([0.046, 0.04, 0.04, 0.04, 0.04, 0.05])
SIG8_0 = 0.81
OM = 0.31


def growth_eps(eps):
    def rhs(N, y):
        a = np.exp(N)
        e2 = OR0 * a ** -4 + OM * a ** -3 + (1 - OM - OR0)
        dlnh = 0.5 * (-4 * OR0 * a ** -4 - 3 * OM * a ** -3) / e2
        om_a = OM * a ** -3 / e2
        return [y[1], -(2.0 + dlnh) * y[1] + 1.5 * om_a * (1 + eps) * y[0]]

    sol = solve_ivp(rhs, (np.log(1 / 1101.0), 0.0), [1e-3, 1e-3],
                    method="DOP853", rtol=1e-10, atol=1e-14,
                    dense_output=True)
    return lambda N: sol.sol(N)[0]


print(f"{'epsilon (pegamento)':>21} {'chi2_fsigma8':>13}  {'veredicto':>24}")
for eps in (-0.30, -0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20, +0.30):
    D = growth_eps(eps)
    fs = []
    for z in RSD_Z:
        N = np.log(1 / (1 + z))
        h = 1e-4
        f = (D(N + h) - D(N - h)) / (2 * h * D(N))
        fs.append(f * SIG8_0 * D(N) / D(0.0))
    fs = np.array(fs)
    c2 = float(np.sum(((fs - RSD_FS8) / RSD_ERR) ** 2))
    if c2 < 12:
        ver = "OK (igual que LCDM)"
    elif c2 < 40:
        ver = "tension"
    else:
        ver = "excluido"
    print(f"{eps:+20.2f} {c2:13.1f}  {ver:>24}")
