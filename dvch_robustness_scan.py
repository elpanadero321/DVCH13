#!/usr/bin/env python3
"""
DVCH robustness scan over (n, beta) parameter space.
Tests background viability conditions across a grid of parameters.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

Omega_m0 = 0.30
Omega_r0 = 9.0e-5
Omega_L0 = 1.0 - Omega_m0 - Omega_r0


def solve_background(n, beta, z_array):
    """Simplified background solver for the scan."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

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

        dOL_dOm = OL_z * n / Om_z if Om_z > 1e-30 else 0.0
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

    z_span = (z_array[0], z_array[-1])
    try:
        sol = solve_ivp(rhs, z_span, [Omega_m0], t_eval=z_array, rtol=1e-8,
                       atol=1e-10, method="RK45", max_step=0.05)
        if not sol.success or sol.y[0].shape[0] != len(z_array):
            return None
    except Exception:
        return None

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

    # Compute Q_tilde
    E = np.sqrt(np.maximum(E2_arr, 0))
    ratio = np.where(Om_arr > 1e-30, OL_arr / Om_arr, 0.0)
    denom_q = 1.0 + n * ratio
    bracket = n - beta * (4.0 * Omega_r0 * (1.0 + z_array)**4 + 3.0 * Om_arr) / (3.0 * (1.0 + beta * E2_arr))
    Qtilde = -E * OL_arr / np.where(denom_q > 0, denom_q, 1e-15) * bracket

    # Transition redshift
    dE2 = np.gradient(E2_arr, z_array)
    q = -1.0 + (1.0 + z_array) / (2.0 * E2_arr) * dE2
    sign_changes = np.where(np.diff(np.sign(q)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        z1, z2 = z_array[idx], z_array[idx + 1]
        q1, q2 = q[idx], q[idx + 1]
        z_t = z1 - q1 * (z2 - z1) / (q2 - q1)
    else:
        z_t = np.nan

    return {
        "E2": E2_arr,
        "Om": Om_arr,
        "OL": OL_arr,
        "Qtilde": Qtilde,
        "z_t": z_t,
    }


def check_viability(res, z_array):
    """Check if a background solution is viable."""
    if res is None:
        return False, "integration_failed"
    E2 = res["E2"]
    Om = res["Om"]
    OL = res["OL"]
    Qt = res["Qtilde"]
    z_t = res["z_t"]

    if np.any(~np.isfinite(E2)):
        return False, "non_finite_E2"
    if np.any(E2 <= 0):
        return False, "E2_nonpositive"
    if np.any(Om < -1e-10):
        return False, "negative_Om"
    if np.any(OL < -1e-10):
        return False, "negative_OL"
    if np.any(~np.isfinite(Qt)):
        return False, "non_finite_Q"
    if np.any(Qt > 0):
        return False, "Q_positive"
    if not (0 < z_t < 2):
        return False, "no_transition"
    return True, "viable"


def main():
    print("=" * 60)
    print("DVCH Robustness Scan")
    print("=" * 60)

    z_scan = np.linspace(0, 5, 200)

    n_values = np.linspace(0.05, 0.45, 16)
    beta_values = np.logspace(-6, -1, 17)

    results = []
    viable_count = 0
    total = len(n_values) * len(beta_values)

    for n_val in n_values:
        for beta_val in beta_values:
            res = solve_background(n_val, beta_val, z_scan)
            viable, reason = check_viability(res, z_scan)
            row = {
                "n": n_val,
                "beta": beta_val,
                "viable": viable,
                "reason": reason,
            }
            if viable and res is not None:
                row["z_t"] = res["z_t"]
                row["Qtilde_max"] = np.max(res["Qtilde"])
                row["Qtilde_min"] = np.min(res["Qtilde"])
                row["E2_min"] = np.min(res["E2"])
                viable_count += 1
            else:
                row["z_t"] = np.nan
                row["Qtilde_max"] = np.nan
                row["Qtilde_min"] = np.nan
                row["E2_min"] = np.nan
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("dvch_robustness_scan.csv", index=False)
    print(f"Grid: {total} points, viable: {viable_count}, fraction: {viable_count/total:.3f}")

    # Summary
    viable_df = df[df["viable"]]
    summary = pd.DataFrame({
        "metric": ["grid_size", "viable_points", "viable_fraction",
                    "n_range", "beta_range", "median_zt", "largest_Qtilde"],
        "value": [total, viable_count, f"{viable_count/total:.3f}",
                  f"0.05-0.45", f"1e-6 to 1e-1",
                  f"{viable_df['z_t'].median():.3f}" if len(viable_df) > 0 else "N/A",
                  f"{viable_df['Qtilde_max'].max():.2e}" if len(viable_df) > 0 else "N/A"],
    })
    summary.to_csv("dvch_robustness_summary.csv", index=False)
    print("Wrote dvch_robustness_scan.csv and dvch_robustness_summary.csv")

    if len(viable_df) > 0:
        print(f"Median z_t: {viable_df['z_t'].median():.3f}")
        print(f"Largest Qtilde: {viable_df['Qtilde_max'].max():.2e}")

    make_figure(df, n_values, beta_values, viable_count, total)


def make_figure(df, n_values, beta_values, viable_count, total):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Viability map
    ax = axes[0]
    grid = np.zeros((len(beta_values), len(n_values)))
    for i, b in enumerate(beta_values):
        for j, nval in enumerate(n_values):
            row = df[(df["beta"] == b) & (df["n"] == nval)]
            if len(row) > 0:
                grid[i, j] = 1 if row["viable"].values[0] else 0

    cmap = ListedColormap(['#FF6B6B', '#4ECDC4'])
    im = ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   extent=[n_values[0], n_values[-1], np.log10(beta_values[0]),
                           np.log10(beta_values[-1])], origin='lower')
    ax.set_xlabel('n')
    ax.set_ylabel(r'$\log_{10}\beta$')
    ax.set_title(f'Viability Map ({viable_count}/{total} viable)')
    # Add colorbar legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4ECDC4', label='Viable'),
                       Patch(facecolor='#FF6B6B', label='Not viable')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # z_t distribution
    ax2 = axes[1]
    viable_df = df[df["viable"]]
    if len(viable_df) > 0:
        sc = ax2.scatter(viable_df["n"], np.log10(viable_df["beta"]),
                        c=viable_df["z_t"], cmap='viridis', s=30, edgecolors='black', lw=0.3)
        plt.colorbar(sc, ax=ax2, label=r'$z_t$')
    ax2.set_xlabel('n')
    ax2.set_ylabel(r'$\log_{10}\beta$')
    ax2.set_title('Transition Redshift Distribution')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('DVCH Parameter-Space Robustness Scan', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/dvch_robustness_scan.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_robustness_scan.png")


if __name__ == "__main__":
    main()
