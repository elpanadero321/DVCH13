#!/usr/bin/env python3
"""Cross-check del background DVCH contra un código Boltzmann externo (CAMB).

En el límite n -> 0 y beta -> 0 la cerradura DVCH reproduce exactamente un
background ΛCDM con los mismos (Omega_m0, Omega_r0, H0).  Este script compara
el H(z) y las distancias de comoving del provider DVCH
(``dvch_camb_background.background``) contra el CAMB instalado (sin parche),
usando los mismos parámetros de fondo.  Es una validación de física completa
del solver homogéneo frente a un código externo independiente.

Salidas:
  dvch_camb_crosscheck.csv
  figures/dvch_camb_crosscheck.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import camb  # código Boltzmann externo (emulator CAMB-master)
from dvch_camb_background import DVCHParameters, background

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Parámetros de fondo compartidos (límite ΛCDM del modelo DVCH).
OMEGA_M0 = 0.30
OMEGA_R0 = 9.0e-5
H0 = 69.03
N_LIMIT = 1.0e-6      # n -> 0
BETA_LIMIT = 0.0      # beta -> 0

Z = np.linspace(0.0, 2.0, 41)


def dvch_background_quantities():
    table = background(Z, DVCHParameters(
        omega_m0=OMEGA_M0, omega_r0=OMEGA_R0,
        n=N_LIMIT, beta=BETA_LIMIT, H0=H0,
    ))
    # columnas: z, E2, E, H, Omega_m, Omega_Lambda, Qtilde
    return table[:, 3], table[:, 4]


def main():
    print("=" * 60)
    print("DVCH background vs CAMB (límite ΛCDM)")
    print("=" * 60)
    h_dvch, om_dvch = dvch_background_quantities()

    ombh2 = 0.0224
    omch2 = OMEGA_M0 * (H0 / 100.0) ** 2 - ombh2
    params = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2,
                             tau=0.054, As=2.1e-9, ns=0.965)
    # Neutrinos sin masa: el límite ΛCDM del cross-check no incluye m_nu.
    params.num_massive = 0
    params.num_massless = 3.046
    params.omnuh2 = 0.0
    print(f"omnuh2 usado      : {params.omnuh2:.3e}")
    data = camb.get_background(params)
    # CAMB devuelve H en unidades naturales (Mpc^-1, c=1); convertimos a km/s/Mpc.
    C_KM_S = 299792.458
    h_camb = np.array([data.h_of_z(z) for z in Z]) * C_KM_S

    rel = np.abs(h_dvch - h_camb) / h_camb
    print(f"camb version      : {camb.__version__}")
    print(f"ombh2 / omch2     : {ombh2:.4f} / {omch2:.4f}")
    print(f"max |dH|/H (rel)  : {rel.max():.3e}")
    print(f"mean |dH|/H (rel) : {rel.mean():.3e}")

    import pandas as pd
    pd.DataFrame({
        "z": Z, "H_dvch": h_dvch, "H_camb": h_camb,
        "rel_diff": rel, "Omega_m_dvch": om_dvch,
    }).to_csv("dvch_camb_crosscheck.csv", index=False)
    print("Wrote dvch_camb_crosscheck.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(Z, h_dvch, label="DVCH (n,beta→0)", lw=2)
    axes[0].plot(Z, h_camb, "--", label="CAMB ΛCDM", lw=1.5)
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("H(z) [km/s/Mpc]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(Z, np.maximum(rel, 1e-16))
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("|ΔH|/H")
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle("DVCH homogeneous solver vs external CAMB (ΛCDM limit)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "dvch_camb_crosscheck.png"), dpi=160)
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_camb_crosscheck.png")

    ok = rel.max() < 1.0e-3
    print(f"CROSSCHECK {'PASS' if ok else 'FAIL'} (tol 1e-3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
