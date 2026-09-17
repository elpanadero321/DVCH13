"""Build the production proposal covariance matrix for the DVCH Planck MCMC.

WHY THE OLD x25 INFLATION WAS REMOVED
------------------------------------
The previous version of this script multiplied the *whole* init covariance by
scale**2 with scale=5 (i.e. variance x25, std x5).  That was motivated by a
claim that "the initial points were ~25 sigma from the center", which is false
for the real production run: the initial points are drawn from the reference
normals in dvch_cobaya_short.yaml and sit close to the mode (e.g. DVCH_beta
~1.2e-4 vs ref 1e-4 +/- 2e-5).

The x25 inflation was actively harmful.  Cobaya proposes all 26 sampled
parameters in a single block, and DVCH_beta has a prior of [0, 0.001].  In the
init covmat DVCH_beta has std ~4.1e-4, which is already a large fraction of
the prior width; after x5 the proposal std became ~2.1e-3, i.e. about twice
the ENTIRE prior width.  Almost every joint proposal therefore landed outside
beta's bounds and was rejected, giving the observed ~5% acceptance (~23 of
~500 steps accepted; see dvch_prod_run.log).

The fix: keep the init covariance as-is (it carries the correct correlations
and per-parameter scales from the calibration), and only *cap* the proposal
std of any parameter that exceeds one quarter of its uniform prior width.
width/4 is a safe maximum for a flat prior -- the optimal random-walk std for
a uniform posterior is width/sqrt(12) ~ 0.29*width, so width/4 is close to
optimal while staying comfortably inside the bounds.

Caps are applied with a symmetric congruence scaling cov = D @ cov @ D.T
(D diagonal, D[j,j] = cap_std_j / std_j, 1 for uncapped params) rather than by
editing diagonal entries in place.  Scaling a row and the matching column by
the same factor multiplies the (j,j) entry by f_j**2 and provably preserves
positive-definiteness, whereas shrinking a diagonal entry alone can destroy it
(e.g. [[0.1, 0.9], [0.9, 1]] is indefinite).
"""

import numpy as np
import yaml

INIT_COVMAT = "dvch_prod_init.covmat"
PRIOR_YAML = "dvch_cobaya_short.yaml"
OUT_COVMAT = "dvch_prod_wide.covmat"
STD_CAP_FRAC = 0.25  # cap proposal std at prior_width / 4


def load_covmat(path):
    with open(path) as fh:
        header = fh.readline().lstrip("#").split()
        rows = [line.split() for line in fh if line.strip()]
    names = [n for n in header if n]
    cov = np.array([[float(x) for x in row] for row in rows], dtype=float)
    if cov.shape != (len(names), len(names)):
        raise ValueError(f"{path}: header has {len(names)} names but matrix is {cov.shape}")
    return names, cov


def prior_widths(path):
    with open(path) as fh:
        info = yaml.safe_load(fh)
    widths = {}
    for name, spec in (info.get("params") or {}).items():
        prior = spec.get("prior") or {}
        if "min" in prior and "max" in prior:
            widths[name] = float(prior["max"]) - float(prior["min"])
    return widths


def main():
    names, cov = load_covmat(INIT_COVMAT)
    widths = prior_widths(PRIOR_YAML)
    n = len(names)

    std = np.sqrt(np.diag(cov))
    caps = np.ones(n)
    for j, name in enumerate(names):
        width = widths.get(name)
        if width is None:
            continue
        cap = STD_CAP_FRAC * width
        if std[j] > cap:
            caps[j] = cap / std[j]
            print(f"capped {name:12s} std {std[j]:.6e} -> {cap:.6e} "
                  f"(prior width {width:.6e}, factor {caps[j]:.6e})")

    cov_new = (caps[:, None] * cov) * caps[None, :]
    eig = np.linalg.eigvalsh(cov_new)
    cond = eig.max() / eig.min()

    print(f"\nn_params = {n}")
    print(f"min eig = {eig.min():.6e}, max eig = {eig.max():.6e}")
    print(f"condition = {cond:.6e}")
    print(f"positive definite = {bool(eig.min() > 0)}")
    if not (np.allclose(cov_new, cov_new.T, atol=1e-12) and eig.min() > 0):
        raise RuntimeError("result is not symmetric positive definite")

    print(f"\n{'param':14s} {'init std':>14s} {'new std':>14s} {'width/4':>14s}  capped")
    new_std = np.sqrt(np.diag(cov_new))
    for j, name in enumerate(names):
        width = widths.get(name)
        cap = STD_CAP_FRAC * width if width is not None else None
        print(f"{name:14s} {std[j]:14.6e} {new_std[j]:14.6e} "
              f"{cap if cap is not None else float('nan'):14.6e}  {caps[j] < 1}")

    with open(OUT_COVMAT, "w") as out:
        out.write("# " + " ".join(names) + "\n")
        for row in cov_new:
            out.write(" ".join(f"{x:.18e}" for x in row) + "\n")
    print(f"\nWrote {OUT_COVMAT}")


if __name__ == "__main__":
    main()
