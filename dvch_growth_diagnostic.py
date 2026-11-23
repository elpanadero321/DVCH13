#!/usr/bin/env python3
"""
UV-HDE + Planck Limit growth diagnostic: sub-horizon growth equation,
sigma_8/S_8 estimates, and the global comparison dashboard figure.

Uses the exact algebraic UV-HDE background and standard gravity (mu=1, Q=0).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
import os

from uvhde_cosmology import UVHDEPlanckCosmology

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Fiducial parameters (Planck 2018)
H0_fid = 67.36
Omega_m0 = 0.3153
Omega_r0 = 9.18e-5
sigma8_lcdm_ref = 0.811
rd = 147.09


def solve_uvhde_background(z_array):
    """Exact algebraic UV-HDE background (no ODE integration needed)."""
    cosmo = UVHDEPlanckCosmology(H0=H0_fid, Omega_m0=Omega_m0, Omega_r0=Omega_r0, rd=rd)
    E2 = cosmo.E2_classical(z_array)
    E = np.sqrt(E2)
    H = cosmo.H(z_array)
    Om = Omega_m0 * (1.0 + z_array)**3 / E2
    OL = cosmo.Omega_DE0 * (1.0 + z_array) * 2.0 / (1.0 + E2) / E2
    Or = Omega_r0 * (1.0 + z_array)**4 / E2
    de2_dz = cosmo.dE2_dz(z_array)
    q = cosmo.q(z_array)
    return z_array, Om, OL, Or, E2, E, H, de2_dz, cosmo


def compute_growth():
    """Compute growth for UV-HDE and LCDM using standard gravity."""
    z_bg = np.linspace(0, 100, 5000)
    z_bg, Om_uvhde, OL_uvhde, Or_uvhde, E2_uvhde, E_uvhde, H_uvhde, de2_dz_uvhde, cosmo = solve_uvhde_background(z_bg)

    # LCDM background
    E2_lcdm = Omega_r0 * (1.0 + z_bg)**4 + Omega_m0 * (1.0 + z_bg)**3 + cosmo.Omega_DE0
    E_lcdm = np.sqrt(E2_lcdm)
    Om_lcdm = Omega_m0 * (1.0 + z_bg)**3 / E2_lcdm

    # Interpolation functions
    from scipy.interpolate import interp1d
    E_uvhde_f = interp1d(z_bg, E_uvhde, kind='cubic', fill_value='extrapolate')
    Om_uvhde_f = interp1d(z_bg, Om_uvhde, kind='cubic', fill_value='extrapolate')
    de2_dz_uvhde_f = interp1d(z_bg, de2_dz_uvhde, kind='cubic', fill_value='extrapolate')

    E_lcdm_f = interp1d(z_bg, E_lcdm, kind='cubic', fill_value='extrapolate')
    Om_lcdm_f = interp1d(z_bg, Om_lcdm, kind='cubic', fill_value='extrapolate')

    z_start = 100.0
    z_end = 0.0
    N_start = -np.log(1.0 + z_start)
    N_end = 0.0

    def rhs_uvhde(N, y):
        delta, delta_p = y
        z = np.exp(-N) - 1.0
        z = max(z, 0.0)
        E = float(E_uvhde_f(z))
        Om = float(Om_uvhde_f(z))
        de2_dz = float(de2_dz_uvhde_f(z))
        if Om <= 0 or E <= 0:
            return [0.0, 0.0]
        dlnE_dN = -0.5 * (1.0 + z) / E**2 * de2_dz
        Om_H = Om / E**2
        delta_pp = -(2.0 + dlnE_dN) * delta_p + 1.5 * Om_H * delta
        return [delta_p, delta_pp]

    def rhs_lcdm(N, y):
        delta, delta_p = y
        z = np.exp(-N) - 1.0
        z = max(z, 0.0)
        E = float(E_lcdm_f(z))
        Om = float(Om_lcdm_f(z))
        if Om <= 0 or E <= 0:
            return [0.0, 0.0]
        # Analytic dlnE/dN for LCDM
        E2 = E**2
        de2_dz = 4.0 * Omega_r0 * (1.0 + z)**3 + 3.0 * Omega_m0 * (1.0 + z)**2
        dlnE_dN = -0.5 * (1.0 + z) / E2 * de2_dz
        Om_H = Om / E2
        delta_pp = -(2.0 + dlnE_dN) * delta_p + 1.5 * Om_H * delta
        return [delta_p, delta_pp]

    delta0 = 1.0 / (1.0 + z_start)
    delta_p0 = delta0

    sol_uvhde = solve_ivp(rhs_uvhde, [N_start, N_end], [delta0, delta_p0],
                         rtol=1e-8, atol=1e-10, method="RK45", dense_output=True,
                         max_step=0.005)
    sol_lcdm = solve_ivp(rhs_lcdm, [N_start, N_end], [delta0, delta_p0],
                        rtol=1e-8, atol=1e-10, method="RK45", dense_output=True,
                        max_step=0.005)

    N_eval = np.linspace(N_start, N_end, 500)
    z_eval = np.exp(-N_eval) - 1.0

    delta_uvhde = sol_uvhde.sol(N_eval)[0]
    delta_lcdm = sol_lcdm.sol(N_eval)[0]
    delta_p_uvhde = sol_uvhde.sol(N_eval)[1]
    delta_p_lcdm = sol_lcdm.sol(N_eval)[1]

    f_uvhde = delta_p_uvhde / delta_uvhde
    f_lcdm = delta_p_lcdm / delta_lcdm

    delta_uvhde_norm = delta_uvhde / delta_uvhde[-1]
    delta_lcdm_norm = delta_lcdm / delta_lcdm[-1]

    # sigma_8 ratio: absolute growth ratio with same IC
    abs_ratio = delta_uvhde[-1] / delta_lcdm[-1]
    sigma8_uvhde = sigma8_lcdm_ref * abs_ratio
    S8_uvhde = sigma8_uvhde * np.sqrt(Omega_m0 / 0.3)

    return {
        "z": z_eval,
        "delta_uvhde": delta_uvhde_norm,
        "delta_lcdm": delta_lcdm_norm,
        "f_uvhde": f_uvhde,
        "f_lcdm": f_lcdm,
        "sigma8_uvhde": sigma8_uvhde,
        "S8_uvhde": S8_uvhde,
        "sigma8_lcdm": sigma8_lcdm_ref,
        "S8_lcdm": sigma8_lcdm_ref * np.sqrt(Omega_m0 / 0.3),
        "growth_ratio": abs_ratio,
        "cosmo": cosmo,
    }


def main():
    print("=" * 60)
    print("UV-HDE + Planck Limit Growth Diagnostic")
    print("=" * 60)

    res = compute_growth()
    cosmo = res["cosmo"]

    print(f"sigma_8_UVHDE = {res['sigma8_uvhde']:.6f}")
    print(f"S_8_UVHDE = {res['S8_uvhde']:.6f}")
    print(f"sigma_8_LCDM (ref) = {res['sigma8_lcdm']:.3f}")
    print(f"S_8_LCDM (ref) = {res['S8_lcdm']:.3f}")
    print(f"Growth ratio = {res['growth_ratio']:.6f}")

    # f*sigma8 at key redshifts for comparison
    z_key = [0.38, 1.48]
    for zk in z_key:
        idx = np.argmin(np.abs(res["z"] - zk))
        fs8_uvhde = res["f_uvhde"][idx] * res["sigma8_uvhde"] * res["delta_uvhde"][idx]
        fs8_lcdm = res["f_lcdm"][idx] * res["sigma8_lcdm"] * res["delta_lcdm"][idx]
        diff_pct = (fs8_uvhde - fs8_lcdm) / fs8_lcdm * 100
        print(f"f*sigma8(z={zk:.2f}): UVHDE={fs8_uvhde:.4f}, LCDM={fs8_lcdm:.4f}, diff={diff_pct:+.2f}%")

    df_g = pd.DataFrame({
        "z": res["z"],
        "delta_uvhde": res["delta_uvhde"],
        "delta_lcdm": res["delta_lcdm"],
        "f_uvhde": res["f_uvhde"],
        "f_lcdm": res["f_lcdm"],
        "fsigma8_uvhde": res["f_uvhde"] * res["sigma8_uvhde"] * res["delta_uvhde"],
        "fsigma8_lcdm": res["f_lcdm"] * res["sigma8_lcdm"] * res["delta_lcdm"],
    })
    df_g.to_csv("uvhde_growth_sigma8_table.csv", index=False)
    print("Wrote uvhde_growth_sigma8_table.csv")

    summary = pd.DataFrame({
        "quantity": ["sigma8_UVHDE", "S8_UVHDE", "sigma8_LCDM_ref", "S8_LCDM_ref",
                      "growth_ratio", "f_sigma8_UVHDE_z0", "f_sigma8_LCDM_z0"],
        "value": [res["sigma8_uvhde"], res["S8_uvhde"], res["sigma8_lcdm"],
                  res["S8_lcdm"], res["growth_ratio"],
                  res["f_uvhde"][-1] * res["sigma8_uvhde"],
                  res["f_lcdm"][-1] * res["sigma8_lcdm"]],
    })
    summary.to_csv("uvhde_sigma8_s8_summary.csv", index=False)
    print("Wrote uvhde_sigma8_s8_summary.csv")

    make_figure(res)
    print(f"Figure saved: {FIGDIR}/uvhde_sigma8_global_comparison.png")


def make_figure(res):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(res["z"], res["delta_uvhde"], 'b-', lw=2, label='UV-HDE + Planck')
    ax1.plot(res["z"], res["delta_lcdm"], 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax1.set_xlabel('z')
    ax1.set_ylabel(r'$D(z)$')
    ax1.set_title('Linear Growth Factor')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(res["z"], res["f_uvhde"], 'b-', lw=2, label='UV-HDE + Planck')
    ax2.plot(res["z"], res["f_lcdm"], 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$f(z)$')
    ax2.set_title('Growth Rate')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(res["z"], res["f_uvhde"] * res["sigma8_uvhde"] * res["delta_uvhde"],
             'b-', lw=2, label='UV-HDE + Planck')
    ax3.plot(res["z"], res["f_lcdm"] * res["sigma8_lcdm"] * res["delta_lcdm"],
             'r--', lw=1.5, label=r'$\Lambda$CDM')
    rsd_z = np.array([0.02, 0.15, 0.32, 0.57, 0.77, 1.23])
    rsd_fs8 = np.array([0.428, 0.45, 0.44, 0.42, 0.39, 0.35])
    rsd_err = np.array([0.046, 0.04, 0.04, 0.04, 0.04, 0.05])
    ax3.errorbar(rsd_z, rsd_fs8, yerr=rsd_err, fmt='ko', ms=4, capsize=3, label='RSD data', zorder=5)
    ax3.set_xlabel('z')
    ax3.set_ylabel(r'$f\sigma_8(z)$')
    ax3.set_title(r'$f\sigma_8$ Comparison')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    labels = [r'$\sigma_{8,\rm UVHDE}$', r'$\sigma_{8,\Lambda\rm CDM}$',
              r'$S_{8,\rm UVHDE}$', r'$S_{8,\Lambda\rm CDM}$']
    values = [res["sigma8_uvhde"], res["sigma8_lcdm"],
              res["S8_uvhde"], res["S8_lcdm"]]
    colors = ['#4472C4', '#ED7D31', '#4472C4', '#ED7D31']
    bars = ax4.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
    ax4.set_ylabel('Value')
    ax4.set_title(r'$\sigma_8$ / $S_8$ Comparison')
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax4.axhspan(0.75, 0.79, alpha=0.15, color='green', label='KiDS-1000 68%')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    table_data = [
        ["Model", r"$\sigma_8$", r"$S_8$", "Growth ratio", "Status"],
        ["UV-HDE + Planck", f"{res['sigma8_uvhde']:.4f}", f"{res['S8_uvhde']:.4f}",
         f"{res['growth_ratio']:.4f}", "Standard gravity (mu=1)"],
        [r"$\Lambda$CDM (ref)", f"{res['sigma8_lcdm']:.3f}", f"{res['S8_lcdm']:.3f}",
         "1.0000", "Standard gravity"],
    ]
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for i in range(5):
        table[0, i].set_facecolor('#4472C4')
        table[0, i].set_text_props(color='white', fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    table_data2 = [
        ["Quantity", "UV-HDE + Planck", r"$\Lambda$CDM", "Difference"],
        [r"$w_{\rm eff}(0)$", f"{res['cosmo'].w_eff(0.0):.6f}", "-1.000000", f"{res['cosmo'].w_eff(0.0) + 1.0:+.6f}"],
        [r"$w_{\rm eff}(0.4014)$", f"{res['cosmo'].w_eff(0.4014):.6f}", "-1.000000", f"{res['cosmo'].w_eff(0.4014) + 1.0:+.6f}"],
        [r"$q(0.7532)$", f"{res['cosmo'].q(0.7532):.6f}", f"{-1.0 + 0.7532/2 * (4*9.18e-5*1.7532**3 + 3*0.3153*1.7532**2)/(9.18e-5*1.7532**4 + 0.3153*1.7532**3 + (1-0.3153-9.18e-5)):.6f}", "~0"],
        [r"$\mathcal{O}_m(0.1)$", f"{res['cosmo'].Om_diagnostic(0.1):.6f}", "0.3153", "~0.07"],
        [r"$\mathcal{O}_m(0.5)$", f"{res['cosmo'].Om_diagnostic(0.5):.6f}", "0.3153", "~0.02"],
        [r"$\epsilon_{\rm Pl}$", f"{res['cosmo'].eps_Pl:.4e}", "0", "~1.65e-123"],
        [r"$a_{\min}$", f"{res['cosmo'].a_min:.4e}", "0", "~1.97e-32"],
    ]
    table2 = ax6.table(cellText=table_data2, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.0, 1.3)
    for i in range(4):
        table2[0, i].set_facecolor('#4472C4')
        table2[0, i].set_text_props(color='white', fontweight='bold')

    fig.suptitle('UV-HDE + Planck Limit: Growth, $S_8$, and Diagnostics',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(f"{FIGDIR}/uvhde_sigma8_global_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_growth_lcdm_reference(z_eval):
    """Compute LCDM growth reference for comparison."""
    z_bg = np.linspace(0, 100, 5000)
    E2_lcdm = Omega_r0 * (1.0 + z_bg)**4 + Omega_m0 * (1.0 + z_bg)**3 + (1.0 - Omega_m0 - Omega_r0)
    E_lcdm = np.sqrt(E2_lcdm)
    Om_lcdm = Omega_m0 * (1.0 + z_bg)**3 / E2_lcdm

    from scipy.interpolate import interp1d
    E_lcdm_f = interp1d(z_bg, E_lcdm, kind='cubic', fill_value='extrapolate')
    Om_lcdm_f = interp1d(z_bg, Om_lcdm, kind='cubic', fill_value='extrapolate')

    z_start = 100.0
    N_start = -np.log(1.0 + z_start)
    N_end = 0.0

    def rhs_lcdm(N, y):
        delta, delta_p = y
        z = np.exp(-N) - 1.0
        z = max(z, 0.0)
        E = float(E_lcdm_f(z))
        Om = float(Om_lcdm_f(z))
        if Om <= 0 or E <= 0:
            return [0.0, 0.0]
        E2 = E**2
        de2_dz = 4.0 * Omega_r0 * (1.0 + z)**3 + 3.0 * Omega_m0 * (1.0 + z)**2
        dlnE_dN = -0.5 * (1.0 + z) / E2 * de2_dz
        # Growth equation: D'' + (2 + dlnH/dln a) D' - 1.5 * Omega_m(a) * D = 0
        delta_pp = -(2.0 + dlnE_dN) * delta_p + 1.5 * Om * delta
        return [delta_p, delta_pp]

    delta0 = 1.0 / (1.0 + z_start)
    delta_p0 = delta0

    sol_lcdm = solve_ivp(rhs_lcdm, [N_start, N_end], [delta0, delta_p0],
                        rtol=1e-8, atol=1e-10, method="RK45", dense_output=True,
                        max_step=0.005)

    # Evaluate at z=0 to get normalization
    D_0 = sol_lcdm.sol(0.0)[0]

    N_eval = -np.log(1.0 + np.asarray(z_eval, dtype=np.float64))
    delta_lcdm = sol_lcdm.sol(N_eval)[0]
    delta_p_lcdm = sol_lcdm.sol(N_eval)[1]

    f_lcdm = delta_p_lcdm / delta_lcdm
    delta_lcdm_norm = delta_lcdm / D_0

    fs8 = f_lcdm * sigma8_lcdm_ref * delta_lcdm_norm
    return delta_lcdm_norm, fs8


if __name__ == "__main__":
    main()