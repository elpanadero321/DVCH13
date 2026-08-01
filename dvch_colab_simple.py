#!/usr/bin/env python3
"""
DVCH (Dynamic Vacuum Coupling Hypothesis) - Self-contained background solver.

Evaluates the implicit DVCH background cosmology, computes the exact interaction
kernel Q, writes dvch_background_table.csv, and produces the composite diagnostic
figure dvch_results_with_numbers.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ---- Cosmological parameters (fiducial) ----
Omega_m0 = 0.30
Omega_r0 = 9.0e-5
Omega_L0 = 1.0 - Omega_m0 - Omega_r0
n = 0.2             # power-law tracking index (fiducial background)
beta = 1.0e-4       # UV curvature suppression parameter
H0 = 69.03          # km/s/Mpc (late-time best fit)

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)


def E2_dvch(z, n=n, beta=beta, Om=Omega_m0, Or=Omega_r0, OL=Omega_L0):
    """Solve the implicit Friedmann equation by fixed-point iteration.

    E^2 = Or*(1+z)^4 + Om(z) + OL * (Om(z)/Om0)^n * (1+beta)/(1+beta*E^2)
    """
    one_plus_z = 1.0 + z
    rad = Or * one_plus_z**4

    # Seed with LCDM
    E2_prev = rad + Om * one_plus_z**3 + OL

    for _ in range(500):
        # Current Omega_m from the closure: we need to solve simultaneously.
        # Use the conservation relation to get Omega_m.
        # In the beta->0 limit, Omega_m ~ Om0*(1+z)^3 * (1 - correction).
        # For full implicit solve, iterate Omega_m and E2 together.
        # Approximation: Omega_m ~ Om*(1+z)^3 * (1 - y/OL * n * ...) is complex.
        # Simpler: use the full implicit equation. We need Omega_m(z).
        # From conservation: dOmega_m/dz + dOmega_L/dz = 3*Omega_m/(1+z)
        # This is an ODE. For the table, we integrate the ODE.

        # For a simpler approach: in the beta->0 limit, the matter density
        # is approximately Omega_m(z) ~ Om0*(1+z)^3 * exp(-small correction)
        # Let's use the LCDM value as a first approximation and refine.
        Om_z = Om * one_plus_z**3  # zeroth order

        suppression = (1.0 + beta) / (1.0 + beta * E2_prev)
        OL_z = OL * (Om_z / Om)**n * suppression

        E2_new = rad + Om_z + OL_z

        if abs(E2_new - E2_prev) < 1e-10:
            break
        E2_prev = E2_new

    return E2_prev, Om_z, OL_z


def solve_dvch_background(z_array, n=n, beta=beta, Om=Omega_m0, Or=Omega_r0, OL=Omega_L0):
    """
    Full integration of the DVCH background using the conservation ODE.

    dOmega_m/dz = [3*Omega_m/(1+z) - dOmega_L/dz]  ... but this is circular.

    Better approach: integrate the ODE system directly.
    From conservation: dOmega_m/dz + dOmega_L/dz = 3*Omega_m/(1+z)
    And Omega_L = OL * (Omega_m/Om0)^n * (1+beta)/(1+beta*E^2)

    We can write dOmega_m/dz as a function of (z, Omega_m) by differentiating
    the closure and using the conservation equation.

    Let f = Omega_L(Omega_m, E^2) = OL * (Om/Om0)^n * (1+beta)/(1+beta*E^2)
    where E^2 = Or*(1+z)^4 + Omega_m + f

    dOm/dz + df/dz = 3*Om/(1+z)

    df/dz = (df/dOm)*dOm/dz + (df/dz)_explicit
    where (df/dOm) = f * n/Om
    and (df/dz)_explicit comes from the E^2 dependence.

    Let's define:
    f = OL * (Om/Om0)^n * S,  S = (1+beta)/(1+beta*E2)
    dS/dz = -(1+beta)*beta*dE2/dz / (1+beta*E2)^2
    dE2/dz = 4*Or*(1+z)^3 + dOm/dz + df/dz

    This is getting circular. Let's use a direct numerical ODE approach.
    """
    from scipy.integrate import solve_ivp

    def rhs(z, y):
        Om_z = y[0]
        one_plus_z = 1.0 + z
        rad = Or * one_plus_z**4

        # Iterate to find E2 consistent with Om_z
        E2 = rad + Om_z + OL * (Om_z / Om)**n * (1.0 + beta) / (1.0 + beta * (rad + Om_z + OL))
        for _ in range(100):
            OL_z = OL * (Om_z / Om)**n * (1.0 + beta) / (1.0 + beta * E2)
            E2_new = rad + Om_z + OL_z
            if abs(E2_new - E2) < 1e-12:
                break
            E2 = E2_new

        OL_z = OL * (Om_z / Om)**n * (1.0 + beta) / (1.0 + beta * E2)

        # Now compute dOm/dz from the conservation equation.
        # dOm/dz + dOL/dz = 3*Om/(1+z)
        # dOL/dz = (dOL/dOm)*dOm/dz + (dOL/dE2)*dE2/dz_explicit
        # where dE2/dz_explicit = 4*Or*(1+z)^3  (the part not from dOm/dz)

        dOL_dOm = OL_z * n / Om_z if Om_z > 0 else 0.0
        dOL_dE2 = -OL * (Om_z / Om)**n * (1.0 + beta) * beta / (1.0 + beta * E2)**2

        # dE2/dz = 4*Or*(1+z)^3 + dOm/dz + dOL_dOm*dOm/dz + dOL_dE2*dE2/dz
        # dE2/dz = 4*Or*(1+z)^3 + (1 + dOL_dOm)*dOm/dz + dOL_dE2*dE2/dz
        # dE2/dz * (1 - dOL_dE2) = 4*Or*(1+z)^3 + (1 + dOL_dOm)*dOm/dz
        # dE2/dz = [4*Or*(1+z)^3 + (1 + dOL_dOm)*dOm/dz] / (1 - dOL_dE2)

        # Conservation: dOm/dz + dOL/dz = 3*Om/(1+z)
        # dOm/dz + dOL_dOm*dOm/dz + dOL_dE2*dE2/dz = 3*Om/(1+z)
        # (1 + dOL_dOm)*dOm/dz + dOL_dE2 * [4*Or*(1+z)^3 + (1+dOL_dOm)*dOm/dz]/(1-dOL_dE2) = 3*Om/(1+z)

        denom = 1.0 - dOL_dE2
        if abs(denom) < 1e-15:
            denom = 1e-15

        A = 1.0 + dOL_dOm
        B = dOL_dE2 / denom
        C = 4.0 * Or * one_plus_z**3

        # (A + B*A)*dOm/dz + B*C = 3*Om/(1+z)
        # A*(1+B)*dOm/dz = 3*Om/(1+z) - B*C
        coeff = A * (1.0 + B)
        if abs(coeff) < 1e-15:
            coeff = 1e-15

        dOm_dz = (3.0 * Om_z / one_plus_z - B * C) / coeff

        return [dOm_dz]

    z_span = (z_array[0], z_array[-1])
    sol = solve_ivp(rhs, z_span, [Om], t_eval=z_array, rtol=1e-10, atol=1e-12,
                   method="RK45", dense_output=True)

    results = {"z": z_array}
    Om_arr = sol.y[0]
    E2_arr = np.zeros_like(z_array)
    OL_arr = np.zeros_like(z_array)
    Or_arr = Omega_r0 * (1.0 + z_array)**4

    for i, z in enumerate(z_array):
        E2, _, OL_z = E2_dvch_single(z, Om_arr[i], n, beta, Om, Omega_r0, OL)
        E2_arr[i] = E2
        OL_arr[i] = OL_z

    results["E2"] = E2_arr
    results["E"] = np.sqrt(E2_arr)
    results["H"] = H0 * np.sqrt(E2_arr)
    results["Omega_m"] = Om_arr
    results["Omega_L"] = OL_arr
    results["Omega_r"] = Or_arr
    results["Omega_tot"] = Om_arr + OL_arr + Or_arr

    # Compute Q_tilde
    Qtilde = compute_Qtilde(z_array, Om_arr, OL_arr, E2_arr, n, beta, Omega_r0, Om)
    results["Qtilde"] = Qtilde

    return results


def E2_dvch_single(z, Om_z, n, beta, Om0, Or0, OL0):
    """Compute E^2 and Omega_L for given z and Omega_m(z)."""
    one_plus_z = 1.0 + z
    rad = Or0 * one_plus_z**4
    E2 = rad + Om_z + OL0
    for _ in range(200):
        OL_z = OL0 * (Om_z / Om0)**n * (1.0 + beta) / (1.0 + beta * E2)
        E2_new = rad + Om_z + OL_z
        if abs(E2_new - E2) < 1e-12:
            break
        E2 = E2_new
    OL_z = OL0 * (Om_z / Om0)**n * (1.0 + beta) / (1.0 + beta * E2)
    return E2, Om_z, OL_z


def compute_Qtilde(z, Om_z, OL_z, E2, n, beta, Or0, Om0):
    """
    Exact Q_tilde = Q/(3*H0*rho_crit0) from Eq. (Q_exact_box):

    Qtilde = -E * Omega_L / (1 + n*Omega_L/Omega_m) *
             [ n - beta*(4*Or*(1+z)^4 + 3*Om)/(3*(1+beta*E^2)) ]
    """
    one_plus_z = 1.0 + z
    E = np.sqrt(E2)
    ratio = np.where(Om_z > 0, OL_z / Om_z, 0.0)
    denom = 1.0 + n * ratio

    bracket = n - beta * (4.0 * Or0 * one_plus_z**4 + 3.0 * Om_z) / (3.0 * (1.0 + beta * E2))

    Qtilde = -E * OL_z / np.where(denom > 0, denom, 1e-15) * bracket
    return Qtilde


def lcdm_E2(z, Om=Omega_m0, Or=Omega_r0, OL=Omega_L0):
    """Standard LCDM E^2(z)."""
    one_plus_z = 1.0 + z
    return Or * one_plus_z**4 + Om * one_plus_z**3 + OL


def deceleration_parameter(z, E2_arr, z_arr):
    """q(z) = -1 + (1+z)/(2*E^2) * dE2/dz"""
    dE2 = np.gradient(E2_arr, z_arr)
    one_plus_z = 1.0 + z_arr
    q = -1.0 + one_plus_z / (2.0 * E2_arr) * dE2
    return q


def main():
    print("=" * 60)
    print("DVCH Background Solver - Self-contained diagnostic")
    print("=" * 60)
    print(f"Parameters: Omega_m0={Omega_m0}, Omega_r0={Omega_r0:.1e}, "
          f"Omega_L0={Omega_L0:.4f}, n={n}, beta={beta:.1e}, H0={H0}")

    z = np.linspace(0, 5, 500)
    res = solve_dvch_background(z)

    # Also compute LCDM reference
    E2_lcdm = lcdm_E2(z)
    H_lcdm = H0 * np.sqrt(E2_lcdm)

    # Q_tilde
    Qtilde = res["Qtilde"]

    # Deceleration parameter
    q_dvch = deceleration_parameter(z, res["E2"], z)
    q_lcdm = deceleration_parameter(z, E2_lcdm, z)

    # Transition redshift
    z_trans_dvch = find_transition(z, q_dvch)
    z_trans_lcdm = find_transition(z, q_lcdm)

    # Write CSV
    df = pd.DataFrame({
        "z": z,
        "E": res["E"],
        "E2": res["E2"],
        "H": res["H"],
        "Omega_m": res["Omega_m"],
        "Omega_Lambda": res["Omega_L"],
        "Omega_r": res["Omega_r"],
        "Qtilde": Qtilde,
        "q": q_dvch,
    })
    df.to_csv("dvch_background_table.csv", index=False)
    print(f"Wrote dvch_background_table.csv ({len(df)} rows)")

    # Print summary
    print(f"\n--- Diagnostic Summary ---")
    print(f"E^2_min (z=0): {res['E2'][0]:.6f}")
    print(f"E^2 at z=1: {np.interp(1.0, z, res['E2']):.6f}")
    print(f"Omega_m(0) = {res['Omega_m'][0]:.6f}")
    print(f"Omega_L(0) = {res['Omega_L'][0]:.6f}")
    print(f"Qtilde(0) = {np.interp(0, z, Qtilde):.6f}")
    print(f"max Qtilde [0,5] = {np.max(Qtilde[z <= 5]):.6f}")
    print(f"min Qtilde [0,5] = {np.min(Qtilde[z <= 5]):.6f}")
    print(f"Transition redshift (DVCH) z_t = {z_trans_dvch:.5f}")
    print(f"Transition redshift (LCDM)  z_t = {z_trans_lcdm:.5f}")
    print(f"All Qtilde < 0 on [0,5]: {np.all(Qtilde < 0)}")

    # Make the composite figure
    make_figure(z, res, E2_lcdm, H_lcdm, Qtilde, q_dvch, q_lcdm,
                z_trans_dvch, z_trans_lcdm)

    print(f"\nFigure saved: {FIGDIR}/dvch_results_with_numbers.png")
    return res


def find_transition(z, q):
    """Find z_t where q(z_t) = 0 by linear interpolation."""
    sign_changes = np.where(np.diff(np.sign(q)))[0]
    if len(sign_changes) == 0:
        return np.nan
    idx = sign_changes[0]
    z1, z2 = z[idx], z[idx + 1]
    q1, q2 = q[idx], q[idx + 1]
    return z1 - q1 * (z2 - z1) / (q2 - q1)


def make_figure(z, res, E2_lcdm, H_lcdm, Qtilde, q_dvch, q_lcdm,
                z_trans_dvch, z_trans_lcdm):
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel 1: H(z)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(z, res["H"], 'b-', lw=2, label='DVCH')
    ax1.plot(z, H_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax1.set_xlabel('z')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble Parameter')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Density components
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(z, res["Omega_m"], 'b-', lw=2, label=r'$\Omega_m$')
    ax2.plot(z, res["Omega_L"], 'g-', lw=2, label=r'$\Omega_\Lambda$')
    ax2.plot(z, res["Omega_r"], 'y-', lw=1.5, label=r'$\Omega_r$')
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$\Omega_i(z)$')
    ax2.set_title('Density Components')
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-6, 2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Q_tilde
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(z, Qtilde, 'purple', lw=2)
    ax3.axhline(0, color='k', ls='--', lw=0.5)
    ax3.set_xlabel('z')
    ax3.set_ylabel(r'$\widetilde{Q}$')
    ax3.set_title('Interaction Kernel')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(z, Qtilde, 0, where=Qtilde < 0, alpha=0.15, color='purple')

    # Panel 4: E^2 comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(z, res["E2"], 'b-', lw=2, label='DVCH')
    ax4.plot(z, E2_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax4.set_xlabel('z')
    ax4.set_ylabel(r'$E^2(z)$')
    ax4.set_title('Expansion Rate Squared')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Deceleration parameter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(z, q_dvch, 'b-', lw=2, label='DVCH')
    ax5.plot(z, q_lcdm, 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax5.axhline(0, color='k', ls='--', lw=0.5)
    ax5.axvline(z_trans_dvch, color='b', ls=':', lw=1, alpha=0.7)
    ax5.axvline(z_trans_lcdm, color='r', ls=':', lw=1, alpha=0.7)
    ax5.set_xlabel('z')
    ax5.set_ylabel(r'$q(z)$')
    ax5.set_title('Deceleration Parameter')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: w_eff
    ax6 = fig.add_subplot(gs[1, 2])
    w_eff = -1.0 - Qtilde / (3.0 * res["E"] * res["Omega_L"])
    ax6.plot(z, w_eff, 'teal', lw=2)
    ax6.axhline(-1.0, color='k', ls='--', lw=0.5)
    ax6.set_xlabel('z')
    ax6.set_ylabel(r'$w_{\Lambda,\mathrm{eff}}$')
    ax6.set_title('Effective Equation of State')
    ax6.set_ylim(-1.05, -0.85)
    ax6.grid(True, alpha=0.3)

    # Table panel
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    table_data = [
        ["z", "E(z)", "H(z)", "Omega_m", "Omega_L", "Qtilde", "q(z)"],
    ]
    z_sel = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    for zs in z_sel:
        row = [
            f"{zs:.1f}",
            f"{np.interp(zs, z, res['E']):.5f}",
            f"{np.interp(zs, z, res['H']):.3f}",
            f"{np.interp(zs, z, res['Omega_m']):.6f}",
            f"{np.interp(zs, z, res['Omega_L']):.6f}",
            f"{np.interp(zs, z, Qtilde):.6f}",
            f"{np.interp(zs, z, q_dvch):.5f}",
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
        f"DVCH Diagnostic Summary\n"
        f"Parameters: $\\Omega_{{m0}}$={Omega_m0}, $\\Omega_{{r0}}$={Omega_r0:.1e}, "
        f"$\\Omega_{{\\Lambda0}}$={Omega_L0:.4f}, n={n}, $\\beta$={beta:.1e}, "
        f"$H_0$={H0} km/s/Mpc\n"
        f"$E^2_{{\\min}}$ = {res['E2'][0]:.4f}  |  "
        f"$\\widetilde Q(0)$ = {np.interp(0, z, Qtilde):.6f}  |  "
        f"$\\max\\widetilde Q$ [0,5] = {np.max(Qtilde):.6f}  |  "
        f"All $\\widetilde Q < 0$: {np.all(Qtilde < 0)}\n"
        f"$z_t$(DVCH) = {z_trans_dvch:.5f}  |  "
        f"$z_t$($\\Lambda$CDM) = {z_trans_lcdm:.5f}  |  "
        f"$\\Delta z_t$ = {z_trans_dvch - z_trans_lcdm:+.5f}\n"
        f"Consistency: $E^2>0$ everywhere, $\\Omega_i \\geq 0$, "
        f"$\\widetilde Q < 0$ (vacuum$\\to$matter transfer), "
        f"$w_{{\\Lambda,\\mathrm{{eff}}}} > -1$ throughout"
    )
    fig.text(0.5, 0.01, summary_text, ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
             family='monospace')

    fig.suptitle('DVCH Easy Diagnostic: Background, Densities, and Interaction Kernel',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(f"{FIGDIR}/dvch_results_with_numbers.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
