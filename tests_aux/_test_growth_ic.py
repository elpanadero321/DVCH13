"""Test fuerte (el que pediste): el sistema de 2 fluidos con a1=a2=0 debe
reproducir growth_standard a ~1e-3. Tambien comprueba que db == dc (bariones
acompanando) y la invariancia de la coaccion hp = -2 delta_c'."""
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, "/mnt/d/DVCH13-1")
import alpha_general_fit as ag

Om = 0.31
Nn = np.log(ag.A_START)
eps = 1e-3

sol = ag.growth_two_fluid(Om, 0.0, 0.0)[0]
Dc = ag.growth_two_fluid(Om, 0.0, 0.0)[1]
Dstd = ag.growth_standard(Om)

print(" a      D2/Di    Dstd/Di   ratio-1     db-dc    f2     fstd")
for a in (0.05, 0.1, 0.2, 0.5, 0.8, 1.0):
    N = np.log(a)
    y = sol.sol(N)
    d2 = y[0]
    f2 = -0.5 * y[2] / d2 * a
    h = 1e-5
    fs = (Dstd(N + h) - Dstd(N - h)) / (2 * h) / Dstd(N) * a
    print(f" {a:.2f}  {d2/eps:8.3f}  {Dstd(N)/eps:8.3f}  "
          f"{d2/Dstd(N)-1:+.2e}  {y[1]-y[0]:+.1e}  {f2:.4f}  {fs:.4f}")

r = (Dc(0.0) / Dc(Nn)) / (Dstd(0.0) / Dstd(Nn)) - 1.0
print(f"\nconsistencia final (a=1): {r:+.2e}  (objetivo < 1e-3)")
