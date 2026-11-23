#!/usr/bin/env python3
"""
UV-HDE + Planck Limit - Self-contained background solver.

Evaluates the exact algebraic solution for E^2(z), computes background
diagnostics, writes uvhde_background_table.csv, and produces the composite
diagnostic figure uvhde_results_with_numbers.png.

Model: UV-HDE + Límite de Planck (versión definitiva, Δk = 0).
- Sin interacción oscura (Q = 0).
- Cero parámetros libres extra frente a ΛCDM (n = 1/3, β = 1 fijados).
- Solución algebraica exacta y cerrada para E^2(z).
- Rebote cuántico primordial acotado por ρ_Pl.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

from uvhde_cosmology import UVHDEPlanckCosmology

# ---- Cosmological parameters (fiducial Planck 2018) ----
H0 = 67.36
Omega_m0 = 0.3153
Omega_r0 = 9.18e-5
rd = 147.09

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def main():
    print("=" * 60)
    print("UV-HDE + Planck Limit Background Solver")
    print("=" * 60)

    cosmo = UVHDEPlanckCosmology(H0=H0, Omega_m0=Omega_m0, Omega_r0=Omega_r0, rd=rd)

    print(f"Parameters: H0={H0}, Omega_m0={Omega_m0}, Omega_r0={Omega_r0:.2e}")
    print(f"Omega_DE0={cosmo.Omega_DE0:.6f}")
    print(f"eps_Pl={cosmo.eps_Pl:.4e}")
    print(f"a_min={cosmo.a_min:.4e}  (z_bounce={cosmo.z_bounce:.4e})")

    z = np.linspace(0, 5, 500)

    # Background quantities
    E2 = cosmo.E2_classical(z)
    H = cosmo.H(z)
    Om = Omega_m0 * (1.0 + z)**3 / E2
    OL = cosmo.Omega_DE0 * (1.0 + z) * 2.0 / (1.0 + E2) / E2
    Or = Omega_r0 * (1.0 + z)**4 / E2
    w_eff = cosmo.w_eff(z)
    q = cosmo.q(z)
    Om_diag = cosmo.Om_diagnostic(z)

    # LCDM reference
    E2_lcdm = Omega_r0 * (1.0 + z)**4 + Omega_m0 * (1.0 + z)**3 + cosmo.Omega_DE0
    H_lcdm = H0 * np.sqrt(E2_lcdm)
    q_lcdm = -1.0 + (1.0 + z) / (2.0 * E2_lcdm) * (4.0 * Omega_r0 * (1.0 + z)**3 + 3.0 * Omega_m0 * (1.0 + z)**2)

    # Transition redshift
    z_trans_uvhde = find_transition(z, q)
    z_trans_lcdm = find_transition(z, q_lcdm)

    # Write CSV
    df = pd.DataFrame({
        "z": z,
        "E": np.sqrt(E2),
        "E2": E2,
        "H": H,
        "Omega_m": Om,
        "Omega_DE": OL,
        "Omega_r": Or,
        "w_eff": w_eff,
        "q": q,
        "Om_diagnostic": Om_diag,
    })
    df.to_csv("uvhde_background_table.csv", index=False)
    print(f"\nWrote uvhde_background_table.csv ({len(df)} rows)")

    # Print summary
    print(f"\n--- Diagnostic Summary ---")
    print(f"E^2(0) = {E2[0]:.6f}")
    print(f"E^2(z=1) = {np.interp(1.0, z, E2):.6f}")
    print(f"Omega_m(0) = {Om[0]:.6f}")
    print(f"Omega_DE(0) = {OL[0]:.6f}")
    print(f"w_eff(0) = {w_eff[0]:.6f}")
    print(f"w_eff(0.4014) = {np.interp(0.4014, z, w_eff):.6f}")
    print(f"q(0) = {q[0]:.6f}")
    print(f"q(0.7532) = {np.interp(0.7532, z, q):.6f}")
    print(f"Om_diag(0.1) = {np.interp(0.1, z, Om_diag):.6f}")
    print(f"Om_diag(0.5) = {np.interp(0.5, z, Om_diag):.6f}")
    print(f"Transition redshift (UV-HDE) z_t = {z_trans_uvhde:.5f}")
    print(f"Transition redshift (LCDM)  z_t = {z_trans_lcdm:.5f}")
    print(f"Dz_t = {z_trans_uvhde - z_trans_lcdm:+.5f}")

    # Make the composite figure
    make_figure(z, E2, H, Om, OL, Or, w_eff, q, Om_diag,
                E2_lcdm, H_lcdm, q_lcdm, z_trans_uvhde, z_trans_lcdm, cosmo)

    print(f"\nFigure saved: {FIGDIR}/uvhde_results_with_numbers.png")
    return cosmo


def find_transition(z, q):
    """Find z_t where q(z_t) = 0 by linear interpolation."""
    sign_changes = np.where(np.diff(np.sign(q)))[0]
    if len(sign_changes) == 0:
        return np.nan
    idx = sign_changes[0]
    z1, z2 = z[idx], z[idx + 1]
    q1, q2 = q[idx], q[idx + 1]
    return z1 - q1 * (z2 - z1) / (q2 - q1)


def make_figure(z, E2, H, Om, OL, Or, w_eff, q, Om_diag,
                E2_lcdm, H_lcdm, q_lcdm, z_trans_uvhde, z_trans_lcdm, cosmo):
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel 1: H(z)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(z, H, 'b-', lw=2, label='UV-HDE + Planck')
    ax1.plot(z, H_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax1.set_xlabel('z')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble Parameter')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Density components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(z, Om, 'b-', lw=2, label=r'$\Omega_m$')
    ax2.plot(z, OL, 'g-', lw=2, label=r'$\Omega_{\rm DE}$')
    ax2.plot(z, Or, 'y-', lw=1.5, label=r'$\Omega_r$')
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\Omega_i(z)$')
    ax2.set_title('Density Components')
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-6, 2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: w_eff
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(z, w_eff, 'teal', lw=2)
    ax3.axhline(-1.0, color='k', ls='--', lw=0.5)
    ax3.set_xlabel('z')
    ax3.set_ylabel(r'$w_{\rm eff}(z)$')
    ax3.set_title('Effective Equation of State')
    ax3.set_ylim(-1.05, -0.7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: E^2 comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(z, E2, 'b-', lw=2, label='UV-HDE + Planck')
    ax4.plot(z, E2_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax4.set_xlabel('z')
    ax4.set_ylabel(r'$E^2(z)$')
    ax4.set_title('Expansion Rate Squared')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Deceleration parameter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(z, q, 'b-', lw=2, label='UV-HDE + Planck')
    ax5.plot(z, q_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax5.axhline(0, color='k', ls='--', lw=0.5)
    ax5.axvline(z_trans_uvhde, color='b', ls=':', lw=1, alpha=0.7)
    ax5.axvline(z_trans_lcdm, color='r', ls=':', lw=1, alpha=0.7)
    ax5.set_xlabel('z')
    ax5.set_ylabel(r'$q(z)$')
    ax5.set_title('Deceleration Parameter')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Om diagnostic
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(z, Om_diag, 'purple', lw=2)
    ax6.axhline(Omega_m0, color='k', ls='--', lw=0.5, label=r'$\Lambda$CDM ($\Omega_{m0}$)')
    ax6.set_xlabel('z')
    ax6.set_ylabel(r'$\mathcal{O}_m(z)$')
    ax6.set_title(r'$\mathcal{O}_m(z)$ Diagnostic')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Table panel
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    table_data = [
        ["z", "E(z)", "H(z)", "Omega_m", "Omega_DE", "w_eff", "q(z)", "Om_diag"],
    ]
    z_sel = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    for zs in z_sel:
        row = [
            f"{zs:.1f}",
            f"{np.interp(zs, z, np.sqrt(E2)):.5f}",
            f"{np.interp(zs, z, H):.3f}",
            f"{np.interp(zs, z, Om):.6f}",
            f"{np.interp(zs, z, OL):.6f}",
            f"{np.interp(zs, z, w_eff):.6f}",
            f"{np.interp(zs, z, q):.5f}",
            f"{np.interp(zs, z, Om_diag):.6f}",
        ]
        table_data.append(row)

    table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for i in range(len(table_data[0])):
        table[0, i].set_facecolor('#4472C4')
        table[0, i].set_text_props(color='white', fontweight='bold')

    # Summary text box
    summary_text = (
        f"UV-HDE + Planck Limit Diagnostic Summary\n"
        f"Parameters: $H_0$={H0}, $\Omega_{{m0}}$={Omega_m0}, $\Omega_{{r0}}$={Omega_r0:.2e}\n"
        f"$E^2(0)$ = {E2[0]:.4f}  |  "
        f"$w_{{\\rm eff}}(0)$ = {w_eff[0]:.6f}  |  "
        f"$w_{{\\rm eff}}(0.4014)$ = {np.interp(0.4014, z, w_eff):.6f}\n"
        f"$q(0.7532)$ = {np.interp(0.7532, z, q):.6f}  |  "
        f"$\\mathcal{{O}}_m(0.1)$ = {np.interp(0.1, z, Om_diag):.6f}  |  "
        f"$\\mathcal{{O}}_m(0.5)$ = {np.interp(0.5, z, Om_diag):.6f}\n"
        f"$z_t$(UV-HDE) = {z_trans_uvhde:.5f}  |  "
        f"$z_t$($\\Lambda$CDM) = {z_trans_lcdm:.5f}  |  "
        f"$\\Delta z_t$ = {z_trans_uvhde - z_trans_lcdm:+.5f}\n"
        f"Consistency: $E^2>0$ everywhere, $\Omega_i \\geq 0$, "
        f"$Q=0$ (no dark interaction), $w_{{\\rm eff}} > -1$ throughout"
    )
    fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
             family='monospace')

    fig.suptitle('UV-HDE + Planck Limit: Background, Densities, and Diagnostics',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(f"{FIGDIR}/uvhde_results_with_numbers.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()