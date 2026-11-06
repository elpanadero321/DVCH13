#!/usr/bin/env python3
"""
DVCH real-data diagnostics: transition redshift, interaction-sign diagnostic,
and regularized statefinder plane.

A diferencia de la version anterior (que usaba la tabla de fondo fiducial),
estos diagnosticos se generan a partir del MEJOR AJUSTE REAL de la seccion
tardia (leido de dvch_late_time_fit_summary.csv), integrando la ecuacion de
conservacion CDM-vacio con los parametros del ajuste. El paper debe citar
exactamente los valores de dvch_realdata_diagnostics.csv.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dvch_vs_lcdm_real as m

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))


def read_bestfit():
    """Lee los parametros del mejor ajuste DVCH del resumen tardio."""
    path = os.path.join(_HERE, "dvch_late_time_fit_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Falta {path}. Corre dvch_vs_lcdm_real.py primero.")
    frame = pd.read_csv(path)
    frame = frame[frame.model == "dvch"]
    params = {}
    for key in ("Om", "H0", "n", "beta"):
        col = f"{key}_median"
        series = frame[[col]].dropna()
        if series.empty:
            raise KeyError(f"Falta {col} en {path}")
        params[key] = float(series.iloc[0, 0])
    return params


def build_background(Om, n, beta, H0, zmax=2.0, nz=401):
    """Fondo DVCH con conservacion (matter integrado, no (1+z)^3)."""
    z = np.linspace(0.0, zmax, nz)
    matter = m._integrate_dvch_matter(Om, n, beta, z)
    if matter is None:
        raise RuntimeError("La integracion de fondo fallo para el best-fit")
    OL = 1.0 - Om - m.Omega_r0
    rows = []
    for zi, mi in zip(z, matter):
        e2, vacuum = m._dvch_closure(float(mi), float(zi), Om, n, beta, OL)
        rad = m.Omega_r0 * (1.0 + zi) ** 4
        e = np.sqrt(e2)
        qtilde = -e * vacuum / (1.0 + n * vacuum / mi) * (
            n - beta * (4.0 * rad + 3.0 * mi) / (3.0 * (1.0 + beta * e2)))
        # q = -1 + (1+z)/(2E^2) dE2/dz; dE2/dz se calcula por diferencias
        rows.append((zi, e2, vacuum / OL, mi, qtilde))
    e2 = np.array([r[1] for r in rows])
    vac = np.array([r[2] for r in rows])
    mat = np.array([r[3] for r in rows])
    qtil = np.array([r[4] for r in rows])
    # dE2/dz por diferencias finitas centradas
    de2 = np.gradient(e2, z)
    q = -1.0 + (1.0 + z) / (2.0 * e2) * de2
    return dict(z=z, E2=e2, E=np.sqrt(e2), Om=mat, OL=vac, Qtilde=qtil, q=q)


def statefinder(z, E2):
    e = np.sqrt(E2)
    dE2 = np.gradient(E2, z)
    d2E2 = np.gradient(dE2, z)
    q = -1.0 + (1.0 + z) / (2.0 * E2) * dE2
    r = (1.0 - (1.0 + z) / E2 * dE2
         + (1.0 + z)**2 / (4.0 * E2**2) * dE2**2
         + (1.0 + z)**2 / (2.0 * E2) * d2E2)
    denom_s = 3.0 * (q + 0.5)
    s = np.where(np.abs(denom_s) > 1e-10, (r - 1.0) / denom_s, np.nan)
    return q, r, s


def transition_redshift(z, q):
    sign_changes = np.where(np.diff(np.sign(q)))[0]
    if len(sign_changes) == 0:
        return np.nan
    idx = sign_changes[0]
    z1, z2 = z[idx], z[idx + 1]
    q1, q2 = q[idx], q[idx + 1]
    return z1 - q1 * (z2 - z1) / (q2 - q1)


def main():
    print("=" * 60)
    print("DVCH Real-Data Diagnostics (mejor ajuste real)")
    print("=" * 60)

    best = read_bestfit()
    print(f"  Best-fit: Om={best['Om']:.4f} H0={best['H0']:.2f} "
          f"n={best['n']:.4f} beta={best['beta']:.4f}")

    bg = build_background(best["Om"], best["n"], best["beta"], best["H0"])
    z = bg["z"]
    q = bg["q"]
    qtil = bg["Qtilde"]

    z_t_dvch = transition_redshift(z, q)

    # LCDM de referencia con el mismo Om
    E2_lcdm = m.E2_lcdm(z, best["Om"])
    dE2_lcdm = np.gradient(E2_lcdm, z)
    q_lcdm = -1.0 + (1.0 + z) / (2.0 * E2_lcdm) * dE2_lcdm
    z_t_lcdm = transition_redshift(z, q_lcdm)

    delta_zt = z_t_dvch - z_t_lcdm

    # Q-sign diagnostic sobre 0 <= z <= 2
    Q_min = float(np.min(qtil))
    Q_max = float(np.max(qtil))

    # Statefinder
    _, r_sf, s_sf = statefinder(z, bg["E2"])
    _, r_l, s_l = statefinder(z, E2_lcdm)

    print(f"z_t (DVCH) = {z_t_dvch:.5f}")
    print(f"z_t (LCDM) = {z_t_lcdm:.5f}")
    print(f"Delta z_t = {delta_zt:+.5f}")
    print(f"Q_min [0,2] = {Q_min:.6f}")
    print(f"Q_max [0,2] = {Q_max:.6f}")

    diag = pd.DataFrame({
        "diagnostic": ["z_t_DVCH", "z_t_LCDM", "Delta_zt",
                       "Q_min", "Q_max",
                       "r_DVCH_z0", "s_DVCH_z0", "r_LCDM_z0", "s_LCDM_z0",
                       "Om_bestfit", "H0_bestfit", "n_bestfit", "beta_bestfit"],
        "value": [f"{z_t_dvch:.5f}", f"{z_t_lcdm:.5f}", f"{delta_zt:+.5f}",
                  f"{Q_min:.6f}", f"{Q_max:.6f}",
                  f"{r_sf[0]:.5f}", f"{s_sf[0]:.5f}",
                  f"{r_l[0]:.5f}", f"{s_l[0]:.5f}",
                  f"{best['Om']:.4f}", f"{best['H0']:.2f}",
                  f"{best['n']:.4f}", f"{best['beta']:.4f}"],
    })
    diag.to_csv("dvch_realdata_diagnostics.csv", index=False)
    print("Wrote dvch_realdata_diagnostics.csv")

    make_figure(z, q, q_lcdm, qtil, z_t_dvch, z_t_lcdm,
                z, r_sf, s_sf, r_l, s_l, Q_min, Q_max)


def make_figure(z, q_dvch, q_lcdm, Qtilde, z_t_dvch, z_t_lcdm,
                z_sf, r_sf, s_sf, r_l, s_l, Q_min, Q_max):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    ax1.plot(z, q_dvch, 'b-', lw=2, label='DVCH')
    ax1.plot(z, q_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax1.axhline(0, color='k', ls='--', lw=0.5)
    ax1.axvline(z_t_dvch, color='b', ls=':', lw=1.5,
                label=f'$z_t^{{\\rm DVCH}}={z_t_dvch:.5f}$')
    ax1.axvline(z_t_lcdm, color='r', ls=':', lw=1.5,
                label=f'$z_t^{{\\Lambda\\rm CDM}}={z_t_lcdm:.5f}$')
    ax1.fill_between(z, 0, q_dvch, where=q_dvch > 0, alpha=0.1, color='blue')
    ax1.set_xlabel('z')
    ax1.set_ylabel('q(z)')
    ax1.set_title(f'Deceleration & Transition ($\\Delta z_t={z_t_dvch-z_t_lcdm:+.5f}$)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)

    ax2 = axes[1]
    ax2.plot(z, Qtilde, 'purple', lw=2)
    ax2.axhline(0, color='k', ls='--', lw=0.5)
    ax2.fill_between(z, Qtilde, 0, where=Qtilde < 0, alpha=0.2, color='purple',
                     label='$Q<0$ (vacuum$\\to$matter)')
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\widetilde{Q}$')
    ax2.set_title(f'Interaction-Sign Diagnostic ($\\min={Q_min:.4f}$, '
                  f'$\\max={Q_max:.4f}$)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    mask_sf = (z_sf >= 0) & (z_sf <= 2) & np.isfinite(s_sf) & np.isfinite(r_sf)
    mask_l = (z >= 0) & (z <= 2) & np.isfinite(s_l) & np.isfinite(r_l)
    ax3.plot(s_sf[mask_sf], r_sf[mask_sf], 'b-', lw=2, label='DVCH')
    ax3.plot(s_l[mask_l], r_l[mask_l], 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax3.plot(0, 1, 'k*', ms=15,
             label=r'$\Lambda$CDM fixed point $(s=0, r=1)$')
    ax3.plot(s_sf[0], r_sf[0], 'bo', ms=8, zorder=5)
    ax3.plot(s_l[0], r_l[0], 'ro', ms=8, zorder=5)
    ax3.set_xlabel('s')
    ax3.set_ylabel('r')
    ax3.set_title('Regularized Statefinder Plane')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('DVCH Real-Data Diagnostics (best-fit): Kinematics, '
                 'Interaction Sign, and Statefinder',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/dvch_realdata_diagnostics.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_realdata_diagnostics.png")


if __name__ == "__main__":
    main()
