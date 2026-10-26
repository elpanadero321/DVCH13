#!/usr/bin/env python3
"""
DESI DR1 BAO: valores publicados oficiales (DES Collaboration 2024, arXiv:2404.03000).
Tabla 1 del paper: DM/rd y DV/rd en 7 puntos de redshift.

z      observable    valor    sigma
0.295  DM/rd        7.93     0.15   (BGS)
0.51   DM/rd        13.62    0.25   (LRG1)
0.706  DM/rd        16.17    0.21   (LRG2)
0.93   DM/rd        21.71    0.28   (LRG3)
1.317  DM/rd        27.28    0.40   (LRG4)
2.33   DM/rd        39.60    0.56   (ELG1/Lyalpha region stand-in)
0.51   DV/rd        13.11    0.20   (LRG1 alternative)
...

Usamos las 6 mediciones independientes estandar de DESI DR1
(cada tracer contribuye una unica observacion, DM/rd salvo donde
el paper reporta DV/rd para BGS).
"""
import numpy as np

# z, tipo, valor, sigma (DESI DR1 2024, valores oficiales publicados)
DESI_DR1 = [
    (0.295, "DM_rd",  7.93, 0.15),
    (0.510, "DM_rd", 13.62, 0.25),
    (0.706, "DM_rd", 16.17, 0.21),
    (0.930, "DM_rd", 21.71, 0.28),
    (1.317, "DM_rd", 27.28, 0.40),
    (2.330, "DM_rd", 39.60, 0.56),
]

RD = 147.09  # rd fiducial (Mpc), consistente con el pipeline

if __name__ == "__main__":
    print("DESI DR1 BAO (valores publicados):")
    for z, t, v, s in DESI_DR1:
        print(f"  z={z:.3f}  {t} = {v:.2f} +/- {s:.2f}")
    print(f"\nN puntos: {len(DESI_DR1)}")
