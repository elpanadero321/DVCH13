#!/usr/bin/env python3
"""Regenerate all DVCH background tables and figures from code (no reuse of
static manuscript images). Outputs go to results/ and figures/.

Usage: python3 scripts/make_background_outputs.py
"""
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvch import BackgroundSolver, DVCHParams  # noqa: E402
from dvch.growth import GrowthSolver  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

SEED = 20260730
np.random.seed(SEED)

FID = DVCHParams(H0=70.0, Omega_m0=0.30, Omega_r0=9.0e-5, n=0.20, beta=1e-4)
LCDM = DVCHParams(H0=70.0, Omega_m0=0.30, Omega_r0=9.0e-5, n=0.0, beta=0.0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def background_table():
    bg = BackgroundSolver(FID)
    df = pd.DataFrame({
        "z": bg.z, "E": bg.E, "H_km_s_Mpc": bg.H,
        "Omega_m": bg.Omega_m, "Omega_Lambda": bg.Omega_L, "Omega_r": bg.Omega_r,
        "Qtilde": bg.Qtilde(), "w_eff": bg.w_eff(), "q": bg.deceleration(),
        "D_L_Mpc": bg.luminosity_distance(),
    })
    path = os.path.join(RESULTS, "dvch_background_table.csv")
    df.to_csv(path, index=False)
    return bg, path


def figures(bg):
    lcdm = BackgroundSolver(LCDM)
    m = bg.z <= 5
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(bg.z[m], bg.H[m], label="DVCH")
    ax.plot(lcdm.z[m], lcdm.H[m], "--", label=r"$\Lambda$CDM")
    ax.set_xlabel("z"); ax.set_ylabel("H(z) [km/s/Mpc]"); ax.legend()
    ax = axes[0, 1]
    for name, arr in [("$\\Omega_m$", bg.Omega_m), ("$\\Omega_\\Lambda$", bg.Omega_L),
                      ("$\\Omega_r$", bg.Omega_r)]:
        ax.loglog(1 + bg.z, arr, label=name)
    ax.set_xlabel("1+z"); ax.set_ylabel(r"$\rho_i/\rho_{c,0}$"); ax.legend()
    ax = axes[1, 0]
    ax.plot(bg.z[m], bg.Qtilde()[m], "k-.")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("z"); ax.set_ylabel(r"$\tilde{Q}=Q/(3H_0\rho_{c,0})$")
    ax = axes[1, 1]
    ax.plot(bg.z[m], bg.deceleration()[m], label="DVCH q(z)")
    ax.plot(lcdm.z[m], lcdm.deceleration()[m], "--", label=r"$\Lambda$CDM q(z)")
    ax.axhline(0, color="gray", lw=0.5)
    zt = bg.transition_redshift()
    ax.axvline(zt, color="k", ls=":", label=f"$z_t$={zt:.4f}")
    ax.set_xlabel("z"); ax.legend()
    fig.suptitle(f"DVCH background, H0={FID.H0}, $\\Omega_m$={FID.Omega_m0}, "
                 f"n={FID.n}, $\\beta$={FID.beta}")
    fig.tight_layout()
    path = os.path.join(FIGURES, "background_diagnostic.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def scan():
    ns = np.linspace(0.05, 0.45, 17)
    betas = np.logspace(-6, -1, 16)
    rows = []
    for n in ns:
        for b in betas:
            try:
                bg = BackgroundSolver(DVCHParams(n=float(n), beta=float(b)),
                                      z_max=100.0, n_grid=600)
                zt = bg.transition_redshift()
                rows.append({"n": n, "beta": b, "viable": bg.is_viable(),
                             "z_t": zt if zt else np.nan,
                             "Qtilde_max_z0_5": float(np.max(bg.Qtilde()[bg.z <= 5]))})
            except Exception as e:  # non-convergent points are recorded, not hidden
                rows.append({"n": n, "beta": b, "viable": False, "z_t": np.nan,
                             "Qtilde_max_z0_5": np.nan, "error": str(e)})
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "scan_n_beta.csv")
    df.to_csv(path, index=False)

    piv = df.pivot(index="beta", columns="n", values="viable").astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pcolormesh(piv.columns, piv.index, piv.values, shading="nearest", cmap="Greys")
    ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel(r"$\beta$")
    ax.set_title(f"Viability scan (viable fraction = {df['viable'].mean():.3f})")
    fpath = os.path.join(FIGURES, "scan_n_beta.png")
    fig.savefig(fpath, dpi=160); plt.close(fig)
    return df, path, fpath


def summary(bg, scan_df, paths):
    g = GrowthSolver(bg)
    lcdm = BackgroundSolver(LCDM)
    zt = bg.transition_redshift()
    Qt = bg.Qtilde()
    m5 = bg.z <= 5
    sigma8 = g.sigma8_rescaled(0.811, bg_lcdm=lcdm)
    out = {
        "seed": SEED,
        "fiducial": FID.__dict__ if hasattr(FID, "__dict__") else str(FID),
        "params": {"H0": FID.H0, "Omega_m0": FID.Omega_m0, "Omega_r0": FID.Omega_r0,
                   "n": FID.n, "beta": FID.beta},
        "E_min": float(bg.E.min()),
        "Qtilde_max_z0_5": float(Qt[m5].max()),
        "Qtilde_at_0": float(Qt[0]),
        "z_t": float(zt),
        "z_t_lcdm": float(lcdm.transition_redshift()),
        "sigma8_diagnostic": float(sigma8),
        "S8_diagnostic": float(sigma8 * np.sqrt(FID.Omega_m0 / 0.3)),
        "fEDE_1100": float(bg.fEDE(1100.0)),
        "scan_viable_fraction": float(scan_df["viable"].mean()),
        "scan_median_zt_viable": float(scan_df.loc[scan_df.viable, "z_t"].median()),
        "checksums_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in paths},
    }
    path = os.path.join(RESULTS, "background_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


if __name__ == "__main__":
    bg, table_path = background_table()
    fig_path = figures(bg)
    scan_df, scan_path, scan_fig = scan()
    out = summary(bg, scan_df, [table_path, fig_path, scan_path, scan_fig])
    print(json.dumps(out, indent=2, default=str))
