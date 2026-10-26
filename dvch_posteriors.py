#!/usr/bin/env python3
# Posteriores PROVISIONALES (NO CONVERGIDAS) de los parametros DVCH
# a partir de las cadenas existentes. Solo para orientacion, NO reportables.
import glob, numpy as np

def load(prefix):
    files = sorted(glob.glob(f'/mnt/d/DVCH13-1/{prefix}.[0-9]*.txt'))
    chains, names = [], None
    for f in files:
        rows = []
        with open(f) as fh:
            for line in fh:
                if line.startswith('#'):
                    names = line.lstrip('#').split(); continue
                rows.append([float(x) for x in line.split()])
        if rows: chains.append(np.array(rows))
    return chains, names

def posterior(chains, names, pname):
    try: j = names.index(pname)
    except ValueError: return None
    vals = np.concatenate([c[:, j] for c in chains])
    w = np.concatenate([c[:, 1] for c in chains])
    mean = np.average(vals, weights=w)
    std = np.sqrt(np.average((vals-mean)**2, weights=w))
    q = [np.percentile(vals, p) for p in [5, 16, 50, 84, 95]]
    return mean, std, q

for prefix, label in [('dvch_prod_widebeta5', 'FINAL (5 cadenas, CONVERGIDA)'),
                      ('backup_r1_0.19/dvch_prod_widebeta', 'VIEJA 4 cadenas (28k)')]:
    chains, names = load(prefix)
    if not chains:
        print(f"{label}: sin cadenas\n"); continue
    print(f"=== {label} ===")
    print(f"  cadenas={len(chains)}, muestras={sum(len(c) for c in chains)}")
    for p in ['DVCH_n', 'DVCH_beta', 'H0', 'ombh2', 'omch2', 'ns', 'tau', 'ln10As']:
        r = posterior(chains, names, p)
        if r:
            m, s, q = r
            print(f"  {p:10s} media={m:10.5f} sigma={s:8.5f} "
                  f"mediana={q[2]:10.5f} [5%,95%]=[{q[0]:.5f}, {q[4]:.5f}]")
    print()

# Prior de DVCH_n y DVCH_beta para comparar
print("PRIORS (de dvch_cobaya_short.yaml):")
print("  DVCH_n:    uniforme [0.01, 0.25]")
print("  DVCH_beta: uniforme [0, 1.0]")
