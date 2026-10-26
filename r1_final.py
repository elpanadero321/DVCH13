#!/usr/bin/env python3
# R-1 por parametro (Gelman-Rubin marginal) sobre las cadenas finales.
import glob, numpy as np
files = sorted(glob.glob('/mnt/d/DVCH13-1/dvch_prod_widebeta5.[0-9]*.txt'))
chains, names = [], None
for f in files:
    with open(f) as fh:
        header, rows = None, []
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith('#'):
                header = line.lstrip('#').split(); continue
            rows.append([float(x) for x in line.split()])
    names = header; chains.append(np.array(rows))
try: i_end = names.index('minuslogprior')
except ValueError: i_end = len(names)
pidx = list(range(2, i_end))
pnames = [names[i] for i in pidx]
print("R-1 por parametro (cadenas finales):")
worst = []
for j, pn in zip(pidx, pnames):
    cols = [c[:, j] for c in chains]
    n = min(len(x) for x in cols)
    v = np.array([x[len(x)-n:] for x in cols])
    use_first = n // 2
    v = v[:, use_first:]
    m, nn = v.shape
    means = v.mean(axis=1); vark = v.var(axis=1, ddof=1)
    W = vark.mean(); B = nn * means.var(ddof=1)
    V = (nn-1)/nn * W + B/nn
    r = np.sqrt(V/W) - 1.0 if W > 0 else np.nan
    worst.append((r, pn))
worst.sort(reverse=True)
for r, pn in worst:
    ok = "OK" if (r == r and r < 0.03) else "ALERTA"
    print(f"  {pn:22s} {r:9.5f}  {ok}")
