#!/usr/bin/env python3
"""
DESI DR1 BAO: valores publicados oficiales (DESI Collaboration 2024,
arXiv:2404.03002, JCAP 02 (2025) 021). Tabla 1 del paper: D_M/r_d y
D_H/r_d (correlacionados) en los trazadores anisotropicos, y D_V/r_d
(isotropico) en BGS y QSO.

  trazador   z_eff   D_M/r_d        D_H/r_d        r / D_V/r_d
  BGS        0.295   --             --             7.93 +/- 0.15
  LRG1       0.510   13.62 +/- 0.25 20.98 +/- 0.61 -0.445
  LRG2       0.706   16.85 +/- 0.32 20.08 +/- 0.60 -0.420
  LRG3+ELG1  0.930   21.71 +/- 0.28 17.88 +/- 0.35 -0.389
  ELG2       1.317   27.79 +/- 0.69 13.82 +/- 0.42 -0.444
  QSO        1.491   --             --             26.07 +/- 0.67
  Lya QSO    2.330   39.71 +/- 0.94  8.52 +/- 0.17 -0.477

En total 12 observables BAO. El valor 147.09 Mpc que se usaba antes
corresponde al r_d de Planck; NO se usa en el ajuste (r_d se marginaliza
con un prior plano) para no inyectar informacion del CMB en un analisis
que se presenta como independiente del CMB.
"""
import numpy as np

# (trazador, z, [(observable, valor, sigma), ...], correlacion DM-DH)
DESI_DR1 = [
    ("BGS", 0.295, [("DV_rd", 7.93, 0.15)], None),
    ("LRG1", 0.510, [("DM_rd", 13.62, 0.25), ("DH_rd", 20.98, 0.61)], -0.445),
    ("LRG2", 0.706, [("DM_rd", 16.85, 0.32), ("DH_rd", 20.08, 0.60)], -0.420),
    ("LRG3+ELG1", 0.930, [("DM_rd", 21.71, 0.28), ("DH_rd", 17.88, 0.35)], -0.389),
    ("ELG2", 1.317, [("DM_rd", 27.79, 0.69), ("DH_rd", 13.82, 0.42)], -0.444),
    ("QSO", 1.491, [("DV_rd", 26.07, 0.67)], None),
    ("LyaQSO", 2.330, [("DM_rd", 39.71, 0.94), ("DH_rd", 8.52, 0.17)], -0.477),
]

# prior plano marginalizado sobre r_d (Mpc); RD solo es de referencia.
RD_PRIOR = (130.0, 180.0)
RD_REFERENCE = 147.09  # r_d de Planck (no se usa en el ajuste)

N_OBS = sum(len(entry[2]) for entry in DESI_DR1)

if __name__ == "__main__":
    print("DESI DR1 BAO (Tabla 1 oficial, arXiv:2404.03002):")
    for tracer, z, obs, r in DESI_DR1:
        obs_str = ", ".join(f"{t} = {v:.2f} +/- {s:.2f}" for t, v, s in obs)
        r_str = f", r = {r}" if r is not None else ""
        print(f"  {tracer:10s} z={z:.3f}  {obs_str}{r_str}")
    print(f"\nN puntos: {len(DESI_DR1)} trazadores, {N_OBS} observables")
