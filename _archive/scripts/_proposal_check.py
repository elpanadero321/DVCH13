"""Is the learned proposal at the random-walk floor, or is there headroom?

Compares dvch_prod.covmat (final learned proposal) with the empirical posterior
covariance of the stationary tail, both globally and in the H0/gal545 block.
Also fits the optimal overall scale: for RWM the optimal acceptance is ~0.234;
the observed 0.273 implies the proposal is slightly too small.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _r1_diagnostic import load, sampled_cols, header_names

root = Path(__file__).resolve().parent
files = sorted(root.glob("dvch_prod.*.txt"))
names = header_names(files[0])
chains = [sampled_cols(load(f), names)[0] for f in files]
keep = [i for i, nm in enumerate(names)
        if nm not in ("weight", "minuslogpost")
        and not nm.startswith("minuslogprior") and not nm.startswith("chi2")]
pnames = [names[i] for i in keep]

tails = np.concatenate([c[int(len(c) * 0.6):] for c in chains], axis=0)
W_emp = np.cov(tails.T)

prop = np.loadtxt(root / "dvch_prod.covmat")
print(f"proposal shape={prop.shape}  empirical shape={W_emp.shape}")

# global scale comparison
ev_prop = np.linalg.eigvalsh(prop)[::-1]
ev_emp = np.linalg.eigvalsh(W_emp)[::-1]
print(f"\neigenvalue ratio (proposal / empirical), top and bottom 5:")
print(f"  top 5:    {np.round(ev_prop[:5] / ev_emp[:5], 3)}")
print(f"  bottom 5: {np.round(ev_prop[-5:] / ev_emp[-5:], 3)}")
r = np.sqrt(np.diag(prop) / np.diag(W_emp))
print(f"\nper-param std ratio sqrt(prop_diag/emp_diag): "
      f"median={np.median(r):.3f} min={r.min():.3f} max={r.max():.3f}")
worst = np.argsort(-np.abs(np.log(r)))[:6]
print("  most-mismatched params:")
for j in worst:
    print(f"    {pnames[j]:>18}  ratio={r[j]:.3f}")

# correlation-structure match: correlate the eigenvectors
_, Vp = np.linalg.eigh(prop)
_, Ve = np.linalg.eigh(W_emp)
# subspace overlap of the top-6 eigenvectors (the degenerate directions)
for k in (3, 6, 10):
    U6p = Vp[:, -k:]
    U6e = Ve[:, -k:]
    ov = np.linalg.svd(U6p.T @ U6e, compute_uv=False)
    print(f"\ntop-{k} eigenspace: principal angles (deg) = "
          f"{np.round(np.degrees(np.arccos(np.clip(ov, -1, 1))), 1)}")

# the H0 / gal545 block
idx = {n: i for i, n in enumerate(pnames)}
sub = ["H0", "gal545_A_100", "gal545_A_217", "gal545_A_143", "gal545_A_143_217"]
si = [idx[s] for s in sub]
print("\ncorrelation matrix of the slow block (empirical):")
C = W_emp[np.ix_(si, si)] / np.sqrt(np.outer(np.diag(W_emp)[si], np.diag(W_emp)[si]))
print("        " + "".join(f"{s[:9]:>10}" for s in sub))
for a, s in enumerate(sub):
    print(f"{s[:9]:>8}" + "".join(f"{C[a, b]:>10.2f}" for b in range(len(sub))))
Cp = prop[np.ix_(si, si)] / np.sqrt(np.outer(np.diag(prop)[si], np.diag(prop)[si]))
print("\nproposal corr in the same block:")
print("        " + "".join(f"{s[:9]:>10}" for s in sub))
for a, s in enumerate(sub):
    print(f"{s[:9]:>8}" + "".join(f"{Cp[a, b]:>10.2f}" for b in range(len(sub))))

# implied R-1 budget
for m in (4, 6, 8):
    C_infl = (1 + np.sqrt(26 / (m - 1))) ** 2
    for tau in (45, 25):
        n_tail = C_infl * tau / 0.01
        print(f"\nm={m} tau={tau}: R-1<0.01 needs n_tail={n_tail:.0f} "
              f"-> n={2*n_tail:.0f}/chain, total={m*2*n_tail/1000:.0f}k samples "
              f"= {m*2*n_tail/360:.0f} h at 360 samples/hr")
