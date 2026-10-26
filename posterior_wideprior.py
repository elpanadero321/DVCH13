#!/usr/bin/env python3
# Posterior de n y beta con prior ENSANCHADO (n en [0.01, 0.99]).
# Responde: ¿el CMB permite n ~ 0.47 o lo excluye?
import glob
import numpy as np

files = sorted(glob.glob('/mnt/d/DVCH13-1/dvch_prod_wideprior.[0-9]*.txt'))
chains, names = [], None
for f in files:
    rows = []
    with open(f) as fh:
        for line in fh:
            if line.startswith('#'):
                names = line.lstrip('#').split()
                continue
            rows.append([float(x) for x in line.split()])
    if rows:
        chains.append(np.array(rows))

i = {nm: k for k, nm in enumerate(names)}
N = min(len(c) for c in chains)

pooled = np.vstack([c[len(c)-N:] for c in chains])
w = pooled[:, 1] * 1.0
w = w / w.sum()

for p in ['DVCH_n', 'DVCH_beta']:
    v = pooled[:, i[p]]
    q = np.percentile(v, [5, 16, 50, 84, 95])
    print(f"{p}: mediana={q[2]:.4f} [16%,84%]=[{q[1]:.4f},{q[3]:.4f}] "
          f"[5%,95%]=[{q[0]:.4f},{q[4]:.4f}] (max={v.max():.4f})")
    # fraccion de muestras con n > 0.3 y n > 0.4
    print(f"   fraccion {p} > 0.30: {(v > 0.3).mean()*100:.2f}%")
    print(f"   fraccion {p} > 0.45: {(v > 0.45).mean()*100:.2f}%")
    print(f"   fraccion {p} < 0.05: {(v < 0.05).mean()*100:.2f}%")
