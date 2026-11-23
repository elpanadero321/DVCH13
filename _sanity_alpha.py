"""Sanity: la conservacion de la familia alfa y la excursion compensada.
Q~ = d(rho~_c)/dlna + 3 rho~_c  (der numerica, definicion exacta).
Se checea que Q~ coincide con la regla q_over_h_of_a del alpha_general_fit.
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/DVCH13-1")
import alpha_general_fit as ag

Om, a1, a2 = 0.315, 0.0, 1.0
for a in (0.3, 0.5, 0.8):
    h = 1e-6
    rc = lambda x: float(ag.rho_c_tilde(x, Om, a1, a2))  # noqa: E731
    dl = (rc(a + h) - rc(a - h)) / (2 * h)
    Q_num = a * dl + 3 * rc(a)
    Q_rule = float(ag.q_over_h_of_a(a, a1, a2))
    print(f"a={a}: Q numerica={Q_num:+.6f}  regla alpha={Q_rule:+.6f} "
          f"diff={abs(Q_num-Q_rule):.2e}")
