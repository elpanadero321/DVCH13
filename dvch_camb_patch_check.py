#!/usr/bin/env python3
"""Verifica el parche DVCH de CAMB: el fondo SI implementa la cerradura.

Historia: la primera version del parche (solo perturbaciones del CDM)
dejaba el fondo identico a LCDM. Tras anadir la integracion de la
ecuacion de conservacion CDM-vacio en dtauda/grho_no_de/derivs/derivsv
(septimo punto de parche), este script verifica que:

  1. H(z) con DVCH_flag=True difiere de LCDM (como debe) y
  2. E^2(z) reproduce la cerradura integrada de forma independiente
     (tolerancia 2e-3, residuos del RK4 + tabla ln(1+z)=0.02).

Corre en WSL2 con el CAMB parcheado:
    source env.sh && python3 dvch_camb_patch_check.py
"""
import os
import sys

import numpy as np

if not os.environ.get("DVCH_CAMB_ROOT"):
    sys.exit("Falta DVCH_CAMB_ROOT (source env.sh).")
sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])

import camb  # noqa: E402


def main():
    print("camb:", camb.__version__, "->", os.path.dirname(camb.__file__))
    base = dict(H0=67.5, ombh2=0.0224, omch2=0.122, tau=0.054,
                As=2e-9, ns=0.965)
    p0 = camb.set_params(**base)
    r0 = camb.get_results(p0)
    zs = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    h0 = np.asarray(r0.h_of_z(zs))

    print("\n=== 1. El fondo cambia con DVCH (deja de ser LCDM) ===")
    worst_change = 0.0
    for n, beta in ((0.2, 1e-4), (0.42, 0.53), (0.5175, 1.1036), (0.8, 1.5)):
        p = camb.set_params(**base, DVCH_flag=True, DVCH_n=n, DVCH_beta=beta)
        r = camb.get_results(p)
        h = np.asarray(r.h_of_z(zs))
        rel = np.max(np.abs(h / h0 - 1.0))
        worst_change = max(worst_change, rel)
        print(f"  n={n:6.4f} beta={beta:7.4f}: max|dH|/H = {rel:.3e}")

    print("\n=== 2. Reproduce la cerradura (verificacion independiente) ===")
    import subprocess
    out = subprocess.run(
        ["python3", "dvch_camb_patch_verify.py"],
        capture_output=True, text=True)
    print(out.stdout.strip())
    if out.returncode not in (0, None):
        print(out.stderr[-2000:])
        sys.exit(1)

    if worst_change > 1e-3:
        print(f"\nOK: el parche de fondo esta activo (max|dH|/H = "
              f"{worst_change:.2e}).")
    else:
        sys.exit("\nFALLO: el fondo del CAMB parcheado sigue siendo LCDM.")


if __name__ == "__main__":
    main()
