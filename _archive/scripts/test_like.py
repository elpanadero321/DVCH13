#!/usr/bin/env python3
# Test rapido de las likelihoods vectorizadas.
import time
import numpy as np
import dvch_vs_lcdm_real as m

sne = m.load_pantheon()
z, mb, dmb, cov, cinv = sne
print(f"SNe cargados: {len(z)}, cov {cov.shape[0]}x{cov.shape[1]}")

for label, args in [("LCDM", ('lcdm', None)),
                    ("DVCH", ('dvch', (0.5, 0.5)))]:
    model, extra = args
    t0 = time.time()
    c_cc = m.chi2_cc(0.31, 70.0, model, extra)
    c_bao = m.chi2_bao(0.31, 70.0, model, extra)
    c_sne = m.chi2_sne(0.31, 70.0, -19.3, model, extra, sne)
    print(f"{label}: CC={c_cc:.2f}  BAO={c_bao:.2f}  SNe={c_sne:.2f}  "
          f"total={c_cc+c_bao+c_sne:.2f}  ({time.time()-t0:.3f}s)")

# consistencia de la integracion vectorizada vs cuadratura
from scipy.integrate import quad
if True:
    Om, H0 = 0.31, 70.0
    z_test = 1.0
    exact, _ = quad(lambda zp: 1.0/np.sqrt(m.E2_lcdm(zp, Om)), 0.0, z_test, limit=200)
    dl_vec = m._Z(Om, H0, 0.0, 'lcdm', None, np.array([z_test]))
    integ_vec = dl_vec[0] / ((1.0 + z_test) * m.c_speed / H0)
    print(f"\nCheck integral z=1: exacta={exact:.6f} vectorizada={integ_vec:.6f}"
          f"  diffrel={abs(integ_vec-exact)/exact*100:.4f}%")
print("OK")
