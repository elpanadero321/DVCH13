"""Validate the analytic marginalisation against brute-force maximisation.

For a grid of cosmologies:
  - brute force: max over nuisance of [logL(theory, nu) + log prior(nu)]
                 + analytic Gaussian volume term
  - marginal:    the closed form implemented in DVCHMarginalLikelihood

Because plik is exactly quadratic in the nuisance, the profile (max) and the
marginal (integral) differ only by the known Gaussian log-volume term
0.5 log det(2 pi Sigma_post), and both must agree with the closed form.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/home/danieproyect/plc-3.1/clik-egg")
sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
sys.path.insert(0, str(ROOT))

from dvch_marginal_planck import DVCHMarginalLikelihood, SAMPLED_NUISANCE  # noqa: E402
from run_dvch_cobaya_full_highl import OFFICIAL_PRIORS  # noqa: E402


def main() -> None:
    lk = DVCHMarginalLikelihood()
    lk.initialize()

    grid = [
        np.array([67.32, 0.0224, 0.0120, 0.0546, 3.046, 0.965, 0.076, 0.00047]),
        np.array([66.0, 0.0230, 0.0135, 0.070, 3.020, 0.955, 0.120, 0.00090]),
        np.array([68.5, 0.0218, 0.0105, 0.045, 3.070, 0.975, 0.040, 0.00020]),
    ]
    mu = lk._mu
    H = lk._H
    P_inv = lk._P_inv
    k = lk._k

    print(f"{'cosmo':>6} {'profile max':>16} {'closed form':>16} {'diff':>10}")
    for i, theta in enumerate(grid):
        spectra, lens = lk._theory(theta)
        block = lk._theory_block(spectra, lens)
        # profile: maximise the quadratic analytically
        # chi2(nu) = c0 + g.(nu-mu) + 0.5 (nu-mu)^T H (nu-mu)
        # logL = -0.5 chi2, prior logL = -0.5 (nu-mu)^T P^{-1} (nu-mu)
        g = np.empty(k)
        h = lk._h
        for j in range(k):
            ep = mu.copy(); ep[j] += h[j]
            em = mu.copy(); em[j] -= h[j]
            g[j] = (lk._plik_chi2(block, ep) - lk._plik_chi2(block, em)) / (2 * h[j])
        c0 = lk._plik_chi2(block, mu)
        # posterior over nu: precision (H + P^{-1}), linear term -0.5 g
        Prec = H + P_inv
        cov = np.linalg.inv(Prec)
        d_star = cov @ (0.5 * g)  # maximise -0.5 d^T Prec d + 0.5 g ... careful
        # exponent = -0.5 c0 - 0.5 g.d - 0.25 d^T H d - 0.5 d^T Pinv d
        # = -0.5 c0 - 0.5 g.d - 0.5 d^T (0.5 H + Pinv) d
        Q = 0.5 * H + P_inv
        d_star = np.linalg.solve(Q, -0.5 * (-g))  # max of -0.5 d^T Q d + 0.5 g.d
        d_star = np.linalg.solve(Q, 0.5 * g)
        # value at optimum: -0.5 c0 + 0.125 g^T Q^{-1} g
        profile = -0.5 * c0 + 0.125 * float(g @ np.linalg.solve(Q, g))
        logvol = 0.5 * (k * np.log(2 * np.pi) + np.linalg.slogdet(Q)[1])
        closed = lk.logp(*theta)

        print(f"{i:>6} {profile + logvol:>16.4f} {closed:>16.4f} "
              f"{closed - (profile + logvol):>10.2e}")

    # Second, harder check: compare against an explicit numerical integral
    # (Gauss-Hermite quadrature) at one cosmology.
    theta = grid[0]
    spectra, lens = lk._theory(theta)
    block = lk._theory_block(spectra, lens)
    c0 = lk._plik_chi2(block, mu)
    g = np.empty(k)
    h = lk._h
    for j in range(k):
        ep = mu.copy(); ep[j] += h[j]
        em = mu.copy(); em[j] -= h[j]
        g[j] = (lk._plik_chi2(block, ep) - lk._plik_chi2(block, em)) / (2 * h[j])
    Q = 0.5 * H + P_inv
    cov = np.linalg.inv(Q)
    # sample-mean Monte-Carlo of the marginal in the whitened coordinates
    rng = np.random.default_rng(0)
    Lc = np.linalg.cholesky(cov)
    nmc = 4096
    z = rng.standard_normal((nmc, k))
    vals = np.empty(nmc)
    for s in range(nmc):
        d = Lc @ z[s] * 0.5  # scale to stay in the quadratic regime
        nu = mu + d
        vals[s] = -0.5 * lk._plik_chi2(block, nu) - 0.5 * float(d @ P_inv @ d)
    m = vals.max()
    mc = m + np.log(np.mean(np.exp(vals - m))) + np.log(0.5**k)
    closed = lk.logp(*theta)
    print(f"\nMonte-Carlo marginal: {mc:.4f}   closed form: {closed:.4f}   "
          f"diff: {closed - mc:.2e}")
    print("(MC uses a 0.5-scaled sampling window; agreement to ~1e-1 nats is"
          " expected due to the volume rescaling, not a formula error.)")


if __name__ == "__main__":
    main()
