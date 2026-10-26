#!/usr/bin/env python3
# Compara el R-1 multivariado del test (con oversampling) vs la cadena principal.
import glob, numpy as np, sys

def analyze(prefix, label):
    files = sorted(glob.glob(f'/mnt/d/DVCH13-1/{prefix}.[0-9]*.txt'))
    if len(files) < 2:
        print(f"{label}: cadenas insuficientes ({len(files)})")
        return None
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
        names = header
        if rows: chains.append(np.array(rows))
    if len(chains) < 2: return None
    try: i_end = names.index('minuslogprior')
    except ValueError: i_end = len(names)
    pidx = list(range(2, i_end))
    n = min(len(c) for c in chains)
    m = len(chains)
    use_first = n // 2
    means = np.zeros((m, len(pidx))); covs = np.zeros((m, len(pidx), len(pidx)))
    for k, c in enumerate(chains):
        v = c[len(c)-n:, pidx][use_first:]
        if len(v) < 10: return None
        means[k] = v.mean(axis=0); covs[k] = np.cov(v.T)
    moc = np.average(covs, axis=0)
    com = np.atleast_2d(np.cov(means.T))
    d = np.sqrt(np.diag(com))
    if (d <= 0).any(): return None
    corr = (com/d).T/d; nmoc = (moc/d).T/d
    L = np.linalg.cholesky(nmoc); Linv = np.linalg.inv(L)
    M = Linv.dot(corr).dot(Linv.T)
    r = max(np.abs(np.linalg.eigvalsh(M)))
    print(f"{label}: N/cadena={n}, R-1 multivariado={r:.4f}")
    return r

r_test = analyze('dvch_test_blocking', 'TEST oversampling  ')
r_main = analyze('dvch_prod_widebeta', 'cadena principal   ')
if r_test and r_main:
    print(f"\nMejora: {r_main:.4f} -> {r_test:.4f} (factor {r_main/r_test:.2f}x)")
