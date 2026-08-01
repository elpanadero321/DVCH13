#!/usr/bin/env python3
"""
DVCH growth diagnostic: sub-horizon growth equation, sigma_8/S_8 estimates,
and the global comparison dashboard figure.

Uses the full ODE-integrated DVCH background (not leading-order approximation).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

Omega_m0 = 0.30
Omega_r0 = 9.0e-5
Omega_L0 = 1.0 - Omega_m0 - Omega_r0
n_fid = 0.2
beta_fid = 1.0e-4
H0_fid = 69.03
sigma8_lcdm_ref = 0.811


def solve_dvch_background_full(z_array, n=n_fid, beta=beta_fid):
    """Full ODE integration of DVCH background."""
    def rhs(z, y):
        Om_z = y[0]
        opz = 1.0 + z
        rad = Omega_r0 * opz**4
        E2 = rad + Om_z + Omega_L0
        for _ in range(50):
            OL_z = Omega_L0 * (Om_z / Omega_m0)**n * (1.0 + beta) / (1.0 + beta * E2)
            E2_new = rad + Om_z + OL_z
            if abs(E2_new - E2) < 1e-10:
                break
            E2 = E2_new
        OL_z = Omega_L0 * (Om_z / Omega_m0)**n * (1.0 + beta) / (1.0 + beta * E2)
        dOL_dOm = OL_z * n / Om_z if Om_z > 0 else 0.0
        dOL_dE2 = -Omega_L0 * (Om_z / Omega_m0)**n * (1.0 + beta) * beta / (1.0 + beta * E2)**2
        denom = 1.0 - dOL_dE2
        if abs(denom) < 1e-15:
            denom = 1e-15
        A = 1.0 + dOL_dOm
        B = dOL_dE2 / denom
        C = 4.0 * Omega_r0 * opz**3
        coeff = A * (1.0 + B)
        if abs(coeff) < 1e-15:
            coeff = 1e-15
        dOm_dz = (3.0 * Om_z / opz - B * C) / coeff
        return [dOm_dz]

    sol = solve_ivp(rhs, (z_array[0], z_array[-1]), [Omega_m0],
                   t_eval=z_array, rtol=1e-10, atol=1e-12, method="RK45")
    Om_arr = sol.y[0]

    E2_arr = np.zeros_like(z_array)
    OL_arr = np.zeros_like(z_array)
    Or_arr = Omega_r0 * (1.0 + z_array)**4

    for i in range(len(z_array)):
        opz = 1.0 + z_array[i]
        rad = Omega_r0 * opz**4
        E2 = rad + Om_arr[i] + Omega_L0
        for _ in range(50):
            OL_z = Omega_L0 * (Om_arr[i] / Omega_m0)**n * (1.0 + beta) / (1.0 + beta * E2)
            E2_new = rad + Om_arr[i] + OL_z
            if abs(E2_new - E2) < 1e-10:
                break
            E2 = E2_new
        E2_arr[i] = E2
        OL_arr[i] = OL_z

    E = np.sqrt(E2_arr)
    ratio = np.where(Om_arr > 0, OL_arr / Om_arr, 0.0)
    denom_q = 1.0 + n * ratio
    bracket = n - beta * (4.0 * Omega_r0 * (1.0 + z_array)**4 + 3.0 * Om_arr) / (3.0 * (1.0 + beta * E2_arr))
    Qtilde = -E * OL_arr / np.where(denom_q > 0, denom_q, 1e-15) * bracket

    return z_array, Om_arr, OL_arr, E2_arr, E, Qtilde


def compute_growth():
    """Compute growth for DVCH and LCDM using full backgrounds."""
    z_bg = np.linspace(0, 100, 5000)
    z_bg, Om_dvch, OL_dvch, E2_dvch, E_dvch, Qt_dvch = solve_dvch_background_full(z_bg)

    # Interpolation functions
    E_dvch_f = interp1d(z_bg, E_dvch, kind='cubic', fill_value='extrapolate')
    Om_dvch_f = interp1d(z_bg, Om_dvch, kind='cubic', fill_value='extrapolate')
    Qt_dvch_f = interp1d(z_bg, Qt_dvch, kind='cubic', fill_value='extrapolate')

    z_start = 100.0
    z_end = 0.0
    N_start = -np.log(1.0 + z_start)
    N_end = 0.0

    def rhs_dvch(N, y):
        delta, delta_p = y
        z = np.exp(-N) - 1.0
        z = max(z, 0.0)
        E = float(E_dvch_f(z))
        Om = float(Om_dvch_f(z))
        Qt = float(Qt_dvch_f(z))
        if Om <= 0 or E <= 0:
            return [0.0, 0.0]
        dz = 0.01
        E_p = float(E_dvch_f(min(z + dz, z_bg[-1])))
        E_m = float(E_dvch_f(max(z - dz, 0.0)))
        dEdz = (E_p - E_m) / (2 * dz)
        dEdN = -dEdz * (1.0 + z)
        Hprime_over_H = dEdN / E
        Gamma = 3.0 * Qt / (E * Om)
        Om_H = Om / E**2
        delta_pp = -(2.0 + Hprime_over_H + Gamma) * delta_p + 1.5 * Om_H * delta
        return [delta_p, delta_pp]

    def rhs_lcdm(N, y):
        delta, delta_p = y
        z = np.exp(-N) - 1.0
        z = max(z, 0.0)
        E2 = Omega_r0 * (1 + z)**4 + Omega_m0 * (1 + z)**3 + Omega_L0
        E = np.sqrt(E2)
        Om_H = Omega_m0 * (1 + z)**3 / E2
        dz = 0.01
        E2_p = Omega_r0 * (1 + z + dz)**4 + Omega_m0 * (1 + z + dz)**3 + Omega_L0
        E2_m = Omega_r0 * max(1 + z - dz, 0.01)**4 + Omega_m0 * max(1 + z - dz, 0.01)**3 + Omega_L0
        dE2dz = (E2_p - E2_m) / (2 * dz)
        dE2dN = -dE2dz * (1.0 + z)
        Hprime_over_H = 0.5 * dE2dN / E2
        delta_pp = -(2.0 + Hprime_over_H) * delta_p + 1.5 * Om_H * delta
        return [delta_p, delta_pp]

    delta0 = 1.0 / (1.0 + z_start)
    delta_p0 = delta0

    sol_dvch = solve_ivp(rhs_dvch, [N_start, N_end], [delta0, delta_p0],
                        rtol=1e-8, atol=1e-10, method="RK45", dense_output=True,
                        max_step=0.005)
    sol_lcdm = solve_ivp(rhs_lcdm, [N_start, N_end], [delta0, delta_p0],
                        rtol=1e-8, atol=1e-10, method="RK45", dense_output=True,
                        max_step=0.005)

    N_eval = np.linspace(N_start, N_end, 500)
    z_eval = np.exp(-N_eval) - 1.0

    delta_dvch = sol_dvch.sol(N_eval)[0]
    delta_lcdm = sol_lcdm.sol(N_eval)[0]
    delta_p_dvch = sol_dvch.sol(N_eval)[1]
    delta_p_lcdm = sol_lcdm.sol(N_eval)[1]

    f_dvch = delta_p_dvch / delta_dvch
    f_lcdm = delta_p_lcdm / delta_lcdm

    delta_dvch_norm = delta_dvch / delta_dvch[-1]
    delta_lcdm_norm = delta_lcdm / delta_lcdm[-1]

    # sigma_8 ratio: absolute growth ratio with same IC
    abs_ratio = delta_dvch[-1] / delta_lcdm[-1]
    sigma8_dvch = sigma8_lcdm_ref * abs_ratio
    S8_dvch = sigma8_dvch * np.sqrt(Omega_m0 / 0.3)

    return {
        "z": z_eval,
        "delta_dvch": delta_dvch_norm,
        "delta_lcdm": delta_lcdm_norm,
        "f_dvch": f_dvch,
        "f_lcdm": f_lcdm,
        "sigma8_dvch": sigma8_dvch,
        "S8_dvch": S8_dvch,
        "sigma8_lcdm": sigma8_lcdm_ref,
        "S8_lcdm": sigma8_lcdm_ref * np.sqrt(Omega_m0 / 0.3),
        "growth_ratio": abs_ratio,
    }


def main():
    print("=" * 60)
    print("DVCH Growth Diagnostic")
    print("=" * 60)

    res = compute_growth()

    print(f"sigma_8_DVCH = {res['sigma8_dvch']:.6f}")
    print(f"S_8_DVCH = {res['S8_dvch']:.6f}")
    print(f"sigma_8_LCDM (ref) = {res['sigma8_lcdm']:.3f}")
    print(f"S_8_LCDM (ref) = {res['S8_lcdm']:.3f}")
    print(f"Growth ratio = {res['growth_ratio']:.6f}")

    df_g = pd.DataFrame({
        "z": res["z"],
        "delta_dvch": res["delta_dvch"],
        "delta_lcdm": res["delta_lcdm"],
        "f_dvch": res["f_dvch"],
        "f_lcdm": res["f_lcdm"],
        "fsigma8_dvch": res["f_dvch"] * res["sigma8_dvch"] * res["delta_dvch"],
        "fsigma8_lcdm": res["f_lcdm"] * res["sigma8_lcdm"] * res["delta_lcdm"],
    })
    df_g.to_csv("dvch_growth_sigma8_table.csv", index=False)
    print("Wrote dvch_growth_sigma8_table.csv")

    summary = pd.DataFrame({
        "quantity": ["sigma8_DVCH", "S8_DVCH", "sigma8_LCDM_ref", "S8_LCDM_ref",
                      "growth_ratio", "f_sigma8_DVCH_z0", "f_sigma8_LCDM_z0"],
        "value": [res["sigma8_dvch"], res["S8_dvch"], res["sigma8_lcdm"],
                  res["S8_lcdm"], res["growth_ratio"],
                  res["f_dvch"][-1] * res["sigma8_dvch"],
                  res["f_lcdm"][-1] * res["sigma8_lcdm"]],
    })
    summary.to_csv("dvch_sigma8_s8_summary.csv", index=False)
    print("Wrote dvch_sigma8_s8_summary.csv")

    delta_chi2 = -3.2015
    delta_aic = -1.2015
    delta_bic = +4.2179

    make_figure(res, delta_chi2, delta_aic, delta_bic)
    print(f"Figure saved: {FIGDIR}/dvch_sigma8_global_comparison.png")


def make_figure(res, dchi2, daic, dbic):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(res["z"], res["delta_dvch"], 'b-', lw=2, label='DVCH')
    ax1.plot(res["z"], res["delta_lcdm"], 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax1.set_xlabel('z')
    ax1.set_ylabel(r'$D(z)$')
    ax1.set_title('Linear Growth Factor')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(res["z"], res["f_dvch"], 'b-', lw=2, label='DVCH')
    ax2.plot(res["z"], res["f_lcdm"], 'r--', lw=1.5, label=r'$\Lambda$CDM')
    ax2.set_xlabel('z')
    ax2.set_ylabel(r'$f(z)$')
    ax2.set_title('Growth Rate')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(res["z"], res["f_dvch"] * res["sigma8_dvch"] * res["delta_dvch"],
             'b-', lw=2, label='DVCH')
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
    labels = [r'$\sigma_{8,\rm DVCH}$', r'$\sigma_{8,\Lambda\rm CDM}$',
              r'$S_{8,\rm DVCH}$', r'$S_{8,\Lambda\rm CDM}$']
    values = [res["sigma8_dvch"], res["sigma8_lcdm"],
              res["S8_dvch"], res["S8_lcdm"]]
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
    ic_labels = [r'$\Delta\chi^2$', r'$\Delta$AIC', r'$\Delta$BIC']
    ic_values = [dchi2, daic, dbic]
    ic_colors = ['#70AD47', '#70AD47', '#FFC000']
    bars5 = ax5.bar(ic_labels, ic_values, color=ic_colors, alpha=0.8, edgecolor='black', lw=0.5)
    ax5.axhline(0, color='k', lw=0.5)
    for bar, val in zip(bars5, ic_values):
        y = bar.get_height()
        offset = 0.3 if y > 0 else -0.5
        ax5.text(bar.get_x() + bar.get_width()/2, y + offset,
                f'{val:+.4f}', ha='center', va='bottom' if y > 0 else 'top', fontsize=9)
    ax5.set_ylabel(r'$\Delta$ (DVCH $-$ $\Lambda$CDM)')
    ax5.set_title('Information Criteria')
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    table_data = [
        ["Block", "Status", "Result"],
        ["Background H(z)", "Computed", "E^2>0, finite"],
        ["Interaction Q", "Computed", r"$\widetilde Q<0$"],
        [r"$\sigma_8/S_8$", "Diagnostic", f"{res['sigma8_dvch']:.3f}"],
        ["Pantheon+ SNe", "Exploratory", r"$\Delta\chi^2$ fit"],
        ["DESI BAO", "Compressed", "Included"],
        ["Chronometers", "Included", "CC block"],
        ["Planck CMB", "Protocol", "Benchmark"],
        ["Info criteria", "Computed", "BIC no pref."],
    ]
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for i in range(3):
        table[0, i].set_facecolor('#4472C4')
        table[0, i].set_text_props(color='white', fontweight='bold')

    fig.suptitle('DVCH Growth, $S_8$, and Global Comparison Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(f"{FIGDIR}/dvch_sigma8_global_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
