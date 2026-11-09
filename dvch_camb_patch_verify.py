#!/usr/bin/env python3
"""Verifica el CAMB parcheado contra el modulo de referencia.

Compara E^2(z) del CAMB (DVCH_flag=True) con la integracion independiente
de la ecuacion de conservacion (Python, DOP853, alta precision) para varios
(n, beta). Sin neutrinos (num_nu_massless=0) para que Omega_r coincida
exactamente con el del modulo de referencia.
"""
import os
import sys

sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
import numpy as np
import camb
from scipy.integrate import solve_ivp


def closure(x, z, om_m0, om_l0, om_r0, n, beta):
    rad = om_r0 * (1.0 + z) ** 4
    base = om_l0 * x ** n
    if beta <= 0:
        e2 = rad + om_m0 * x + base
        return e2, base
    A = base * (1.0 + beta)
    B = rad + om_m0 * x
    bq = 1.0 - beta * B
    e2 = (-bq + np.sqrt(bq * bq + 4.0 * beta * (A + B))) / (2.0 * beta)
    vac = A / (1.0 + beta * e2)
    # Variante de desacoplamiento temprano (env: DVCH_ZC, DVCH_P):
    # vac = 1 + S*(vac-1), cerradura cuadratica exacta.
    zc = float(os.environ.get("DVCH_ZC", "1e6"))
    psw = float(os.environ.get("DVCH_P", "10"))
    z = float(z)
    S = 1.0 / (1.0 + ((1.0 + z) / (1.0 + zc)) ** psw)
    if S < 1.0 - 1e-14:
        C = rad + om_m0 * x + om_l0 * (1.0 - S)
        K = S * A
        bq = 1.0 - beta * C
        e2 = (-bq + np.sqrt(max(bq * bq + 4.0 * beta * (C + K), 0.0))) / (2.0 * beta)
        vac = om_l0 + S * (A / (1.0 + beta * e2) - om_l0)
    return e2, vac


def reference_e2(zs, om_m0, om_l0, om_r0, n, beta):
    def matter_eq(z, m):
        x = m / om_m0
        e2, vac = closure(x, z, om_m0, om_l0, om_r0, n, beta)
        dvac_dm = vac * n / m
        dvac_de2 = -om_l0 * x ** n * (1.0 + beta) * beta / (1.0 + beta * e2) ** 2
        # switch: la desviacion del vacio (y sus derivadas) escala por S
        zc = float(os.environ.get("DVCH_ZC", "1e6"))
        psw = float(os.environ.get("DVCH_P", "10"))
        Sv = 1.0 / (1.0 + ((1.0 + z) / (1.0 + zc)) ** psw)
        dvac_dm = Sv * dvac_dm
        dvac_de2 = Sv * dvac_de2
        denom = 1.0 - dvac_de2
        coeff = (1.0 + dvac_dm) * (1.0 + dvac_de2 / denom)
        explicit = 4.0 * om_r0 * (1.0 + z) ** 3
        return (3.0 * m / (1.0 + z) - (dvac_de2 / denom) * explicit) / coeff

    zs = np.atleast_1d(zs)
    sol = solve_ivp(matter_eq, (0.0, max(zs.max(), 1e-9)), [om_m0],
                    t_eval=np.sort(zs), rtol=1e-11, atol=1e-14,
                    method="DOP853")
    ms = sol.y[0]
    out = []
    for z, m in zip(np.sort(zs), ms):
        e2, _ = closure(m / om_m0, z, om_m0, om_l0, om_r0, n, beta)
        out.append(e2)
    return np.array(out)


def make_params(**extra):
    p = camb.set_params(H0=67.5, ombh2=0.0224, omch2=0.122, tau=0.054,
                        As=2e-9, ns=0.965)
    p.set_cosmology(H0=67.5, ombh2=0.0224, omch2=0.122,
                    num_massive_neutrinos=0, mnu=0.0,
                    neutrino_hierarchy="degenerate")
    for k, v in extra.items():
        setattr(p, k, v)
    return p


def main():
    r0 = camb.get_results(make_params())
    h = 67.5 / 100.0
    om_m0 = (0.0224 + 0.122) / h**2
    om_l0 = float(r0.get_Omega("de"))
    om_r0 = 1.0 - om_m0 - om_l0
    print(f"Om_m0={om_m0:.6f} Om_L0={om_l0:.6f} Om_r0={om_r0:.6e}")

    zs = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0])
    worst = 0.0
    for n, beta in ((0.2, 1e-4), (0.42, 0.53), (0.5175, 1.1036), (0.8, 1.5)):
        p = make_params(DVCH_flag=True, DVCH_n=n, DVCH_beta=beta)
        r = camb.get_results(p)
        hz = np.array(r.h_of_z(zs))
        e2_camb = (hz / hz[0])**2
        e2_ref = reference_e2(zs, om_m0, om_l0, om_r0, n, beta)
        rel = np.max(np.abs(e2_camb / e2_ref - 1.0))
        worst = max(worst, rel)
        print(f"  n={n:6.4f} beta={beta:7.4f}: max|dE2|/E2 = {rel:.3e} "
              f"(E2(1)={e2_camb[5]:.5f} vs ref {e2_ref[5]:.5f})")

    print(f"\nPeor desviacion: {worst:.2e}")
    if worst < 2e-3:
        print("VERIFICACION OK: el fondo de CAMB reproduce la cerradura DVCH")
        print("(residuos del RK4 + tabla con paso ln(1+z)=0.02).")
    else:
        print("VERIFICACION FALLIDA: el fondo de CAMB no coincide")
        sys.exit(1)


if __name__ == "__main__":
    main()
