#!/usr/bin/env python3
"""
DVCH real-data diagnostics: transition redshift, interaction-sign diagnostic,
and regularized statefinder plane.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def load_background():
    df = pd.read_csv("dvch_background_table.csv")
    return df


def statefinder(df_bg):
    """
    Compute statefinder pair {r, s} for DVCH and LCDM.
    r = \dddot{a}/a / H^3,  s = 1 - 3*r/(r + 1/2 * dH^2/da * ...)
    Standard: r = q + 2*q^2 + (1+z)/(2*E^2) * dE2/dz * (1 + (1+z)/(2*E^2)*dE2/dz)
    Simpler: r = 1 - (1+z)/E^2 * dE2/dz + (1+z)^2/(4*E^4) * (dE2/dz)^2 + (1+z)^2/(2*E^2) * d2E2/dz2
    s = (r - 1) / (3*(q + 1/2))
    """
    z = df_bg["z"].values
    E2 = df_bg["E2"].values
    E = np.sqrt(E2)
    opz = 1.0 + z

    dE2 = np.gradient(E2, z)
    d2E2 = np.gradient(dE2, z)

    q = -1.0 + opz / (2.0 * E2) * dE2

    # Statefinder r
    r = (1.0 - opz / E2 * dE2
         + opz**2 / (4.0 * E2**2) * dE2**2
         + opz**2 / (2.0 * E2) * d2E2)

    # Statefinder s
    denom_s = 3.0 * (q + 0.5)
    s = np.where(np.abs(denom_s) > 1e-10, (r - 1.0) / denom_s, np.nan)

    return z, r, s, q


def main():
    print("=" * 60)
    print("DVCH Real-Data Diagnostics")
    print("=" * 60)

    df_bg = load_background()
    z = df_bg["z"].values
    E2 = df_bg["E2"].values
    E = df_bg["E"].values
    Qtilde = df_bg["Qtilde"].values
    q = df_bg["q"].values

    # Transition redshift
    sign_changes = np.where(np.diff(np.sign(q)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        z1, z2 = z[idx], z[idx + 1]
        q1, q2 = q[idx], q[idx + 1]
        z_t_dvch = z1 - q1 * (z2 - z1) / (q2 - q1)
    else:
        z_t_dvch = np.nan

    # LCDM transition
    Omega_m0 = 0.30
    Omega_r0 = 9.0e-5
    Omega_L0 = 1.0 - Omega_m0 - Omega_r0
    E2_lcdm = Omega_r0 * (1 + z)**4 + Omega_m0 * (1 + z)**3 + Omega_L0
    dE2_lcdm = np.gradient(E2_lcdm, z)
    q_lcdm = -1.0 + (1.0 + z) / (2.0 * E2_lcdm) * dE2_lcdm
    sign_changes_lcdm = np.where(np.diff(np.sign(q_lcdm)))[0]
    if len(sign_changes_lcdm) > 0:
        idx = sign_changes_lcdm[0]
        z1, z2 = z[idx], z[idx + 1]
        q1, q2 = q_lcdm[idx], q_lcdm[idx + 1]
        z_t_lcdm = z1 - q1 * (z2 - z1) / (q2 - q1)
    else:
        z_t_lcdm = np.nan

    delta_zt = z_t_dvch - z_t_lcdm

    # Q-sign diagnostic over -0.95 <= z <= 2
    z_mask = (z >= -0.95) & (z <= 2.0)
    Q_masked = Qtilde[z_mask]
    Q_min = np.min(Q_masked)
    Q_max = np.max(Q_masked)

    # Statefinder
    z_sf, r_sf, s_sf, q_sf = statefinder(df_bg)

    # LCDM statefinder
    dE2_l = np.gradient(E2_lcdm, z)
    d2E2_l = np.gradient(dE2_l, z)
    q_l = -1.0 + (1.0 + z) / (2.0 * E2_lcdm) * dE2_l
    r_l = (1.0 - (1.0 + z) / E2_lcdm * dE2_l
           + (1.0 + z)**2 / (4.0 * E2_lcdm**2) * dE2_l**2
           + (1.0 + z)**2 / (2.0 * E2_lcdm) * d2E2_l)
    denom_s_l = 3.0 * (q_l + 0.5)
    s_l = np.where(np.abs(denom_s_l) > 1e-10, (r_l - 1.0) / denom_s_l, np.nan)

    print(f"z_t (DVCH) = {z_t_dvch:.5f}")
    print(f"z_t (LCDM) = {z_t_lcdm:.5f}")
    print(f"Delta z_t = {delta_zt:+.5f}")
    print(f"Q_min [-0.95,2] = {Q_min:.6f}")
    print(f"Q_max [-0.95,2] = {Q_max:.6f}")

    # Write diagnostics table
    diag = pd.DataFrame({
        "diagnostic": ["z_t_DVCH", "z_t_LCDM", "Delta_zt",
                       "Q_min", "Q_max",
                       "r_DVCH_z0", "s_DVCH_z0", "r_LCDM_z0", "s_LCDM_z0"],
        "value": [f"{z_t_dvch:.5f}", f"{z_t_lcdm:.5f}", f"{delta_zt:+.5f}",
                  f"{Q_min:.6f}", f"{Q_max:.6f}",
                  f"{r_sf[0]:.5f}", f"{s_sf[0]:.5f}",
                  f"{r_l[0]:.5f}", f"{s_l[0]:.5f}"],
    })
    diag.to_csv("dvch_realdata_diagnostics.csv", index=False)
    print("Wrote dvch_realdata_diagnostics.csv")

    make_figure(z, q, q_lcdm, Qtilde, z_t_dvch, z_t_lcdm,
                z_sf, r_sf, s_sf, r_l, s_l, Q_min, Q_max)


def make_figure(z, q_dvch, q_lcdm, Qtilde, z_t_dvch, z_t_lcdm,
                z_sf, r_sf, s_sf, r_l, s_l, Q_min, Q_max):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Deceleration parameter and transition redshift
    ax1 = axes[0]
    ax1.plot(z, q_dvch, 'b-', lw=2, label='DVCH')
    ax1.plot(z, q_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax1.axhline(0, color='k', ls='--', lw=0.5)
    ax1.axvline(z_t_dvch, color='b', ls=':', lw=1.5, label=f'$z_t^{{\\rm DVCH}}={z_t_dvch:.5f}$')
    ax1.axvline(z_t_lcdm, color='r', ls=':', lw=1.5, label=f'$z_t^{{\\Lambda\\rm CDM}}={z_t_lcdm:.5f}$')
    ax1.fill_between(z, 0, q_dvch, where=q_dvch > 0, alpha=0.1, color='blue')
    ax1.set_xlabel('z')
    ax1.set_ylabel('q(z)')
    ax1.set_title(f'Deceleration & Transition ($\\Delta z_t={z_t_dvch-z_t_lcdm:+.5f}$)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 3)

    # Panel 2: Interaction-sign diagnostic
    ax2 = axes[1]
    z_mask = (z >= -0.95) & (z <= 2.0)
    ax2.plot(z[z_mask], Qtilde[z_mask], 'purple', lw=2)
    ax2.axhline(0, color='k', ls='--', lw=0.5)
    ax2.fill_between(z[z_mask], Qtilde[z_mask], 0,
                     where=Qtilde[z_mask] < 0, alpha=0.2, color='purple', label='$Q<0$ (vacuum$\\to$matter)')
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\widetilde{Q}$')
    ax2.set_title(f'Interaction-Sign Diagnostic ($\\min={Q_min:.4f}$, $\\max={Q_max:.4f}$)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Statefinder plane
    ax3 = axes[2]
    mask_sf = (z_sf >= 0) & (z_sf <= 3) & np.isfinite(s_sf) & np.isfinite(r_sf)
    mask_l = (z >= 0) & (z <= 3) & np.isfinite(s_l) & np.isfinite(r_l)
    ax3.plot(s_sf[mask_sf], r_sf[mask_sf], 'b-', lw=2, label='DVCH')
    ax3.plot(s_l[mask_l], r_l[mask_l], 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax3.plot(0, 1, 'k*', ms=15, label=r'$\Lambda$CDM fixed point $(s=0, r=1)$')
    # Mark z=0
    ax3.plot(s_sf[0], r_sf[0], 'bo', ms=8, zorder=5)
    ax3.plot(s_l[0], r_l[0], 'ro', ms=8, zorder=5)
    ax3.set_xlabel('s')
    ax3.set_ylabel('r')
    ax3.set_title('Regularized Statefinder Plane')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('DVCH Real-Data Diagnostics: Kinematics, Interaction Sign, and Statefinder',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/dvch_realdata_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_realdata_diagnostics.png")


if __name__ == "__main__":
    main()
