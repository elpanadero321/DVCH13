#!/usr/bin/env python3
"""Generalizacion del modelo alpha:  Delta(a) = a1*(1-a) + a2*(1-a)^2.

Motivo
------
PENDIENTES_ALPHA.md e INSTRUCCIONES_PARA_IA.md proponen generalizar
Delta = alpha*(1-a)^2 a un analogo de CPL con dos parametros. La regla
general, valida para CUALQUIER desviacion Delta(a), es

    delta_rho_c     = -(a/3) dDelta/da
    delta_rho_L     = Delta - delta_rho_c
    Q/(H rho_crit0) = 3 delta_rho_c + a d(delta_rho_c)/da

con  rho_c = Oc a^-3 + delta_rho_c ,  rho_L = OL + delta_rho_L  y
E2 = Or a^-4 + Om a^-3 + OL + Delta(a).

Para Delta = a1(1-a) + a2(1-a)^2:

    delta_rho_c     = (a/3)[a1 + 2 a2 (1-a)]
    Q/(H rho_crit0) = a[(4/3) a1 + (2/3) a2 (4-5a)]
    w0 = -1 + a1/(3 OL)          (exacto en a=1)

Observacion estructural (el motivo de este script): para cualquier Delta(a)
el fondo es EXACTAMENTE energia oscura desacoplada con rho_DE = OL + Delta(a).
Luego esta generalizacion NO es fisica nueva a nivel de fondo: es otra familia
de dos parametros de w(a). Lo unico que la distingue es como Q entra en la
ecuacion de delta_c. Este script mide las dos cosas por separado.

Uso:
    python3 alpha_general_fit.py                 # verificacion + ajustes
    python3 alpha_general_fit.py --verify-only
    python3 alpha_general_fit.py --maxfev 3000 --starts 4
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dvch_vs_lcdm_real as dvr  # noqa: E402

OR0 = dvr.Omega_r0
FB = 0.156          # fraccion barionica (igual que alpha_model_fit.py)
A_START = 0.02      # inicio de la integracion de crecimiento (z=49)


# ------------------------------------------------------------------
# 1. Cerradura general (forma cerrada)
# ------------------------------------------------------------------
def delta_of_a(a, a1, a2):
    """Delta(a) = a1(1-a) + a2(1-a)^2.

    Sin cast a float: tiene que admitir complejos (derivada por paso
    complejo en _dlna).
    """
    x = 1.0 - np.asarray(a)
    return a1 * x + a2 * x * x


def ddelta_da(a, a1, a2):
    """dDelta/da = -a1 - 2 a2 (1-a)."""
    return -a1 - 2.0 * a2 * (1.0 - np.asarray(a, dtype=float))


def drho_c_of_a(a, a1, a2):
    """delta_rho_c = -(a/3) dDelta/da = (a/3)[a1 + 2 a2 (1-a)]."""
    a = np.asarray(a)
    return (a / 3.0) * (a1 + 2.0 * a2 * (1.0 - a))


def drho_l_of_a(a, a1, a2):
    """delta_rho_L = Delta - delta_rho_c."""
    return delta_of_a(a, a1, a2) - drho_c_of_a(a, a1, a2)


def q_over_h_of_a(a, a1, a2):
    """Q/(H rho_crit0) = 3 delta_rho_c + a d(delta_rho_c)/da.

    Convencion: d rho_c/dt + 3H rho_c = +Q ,  d rho_L/dt = -Q.
    """
    a = np.asarray(a, dtype=float)
    return (4.0 / 3.0) * a1 * a + (2.0 / 3.0) * a2 * a * (4.0 - 5.0 * a)


def rho_c_tilde(a, Om, a1, a2, fb=FB):
    """rho_c / rho_crit0 (solo CDM; los bariones van aparte)."""
    a = np.asarray(a)
    return Om * (1.0 - fb) * a ** -3 + drho_c_of_a(a, a1, a2)


def rho_l_tilde(a, Om, a1, a2):
    """rho_L / rho_crit0."""
    return (1.0 - Om - OR0) + drho_l_of_a(a, a1, a2)


def e2_of_a(a, Om, a1, a2):
    a = np.asarray(a, dtype=float)
    return (OR0 * a ** -4 + Om * a ** -3 + (1.0 - Om - OR0)
            + delta_of_a(a, a1, a2))


def e2_of_z(z, Om, a1, a2):
    return e2_of_a(1.0 / (1.0 + np.asarray(z, dtype=float)), Om, a1, a2)


def w_eff(a, Om, a1, a2):
    """w efectivo de rho_DE = OL + Delta(a): w = -1 - (1/3) dln rho_DE/dln a."""
    a = np.asarray(a, dtype=float)
    rho = (1.0 - Om - OR0) + delta_of_a(a, a1, a2)
    return -1.0 - a * ddelta_da(a, a1, a2) / (3.0 * rho)


def _dlna(f, a, h=1e-20):
    """a*df/da por paso complejo (sin cancelacion: error a precision maquina)."""
    return np.imag(f(a * np.exp(1j * h))) / h


# ------------------------------------------------------------------
# 2. Verificacion: forma cerrada vs ODE independiente + mutaciones
# ------------------------------------------------------------------
def verify(verbose=True):
    om = 0.315
    oc = om * (1 - FB)
    a_all = np.geomspace(1e-4, 1.0, 4001)
    a_q = np.geomspace(1e-2, 1.0, 2001)      # Q/H ~ O(a); a>=1e-2 resuelve

    if verbose:
        print("=" * 78)
        print("VERIFICACION  Delta = a1(1-a) + a2(1-a)^2   (a = 1/(1+z))")
        print("=" * 78)

    cases = [(-0.16, 0.0), (0.0, 0.0), (0.253, -0.41), (0.597, 0.0),
             (-0.3, 0.5), (0.3, -0.8), (0.8, -1.2)]

    for a1, a2 in cases:
        fc = lambda x: rho_c_tilde(x, om, a1, a2)           # noqa: E731
        fl = lambda x: rho_l_tilde(x, om, a1, a2)           # noqa: E731
        qh = q_over_h_of_a(a_q, a1, a2)

        # (a) Friedmann: suma de densidades == E2
        total = (OR0 * a_all ** -4 + om * FB * a_all ** -3 + fc(a_all)
                 + fl(a_all))
        fr = float(np.max(np.abs(total / e2_of_a(a_all, om, a1, a2) - 1.0)))

        # (b) conservacion contra la forma cerrada (paso complejo)
        err_c = float(np.max(np.abs(_dlna(fc, a_q) + 3.0 * fc(a_q) - qh)))
        err_l = float(np.max(np.abs(_dlna(fl, a_q) + qh)))

        # (c) ODE independiente: a d rho_c/da + 3 rho_c = Q/H ; a d rho_L/da = -Q/H
        rho_c1 = oc + drho_c_of_a(1.0, a1, a2)
        rho_l1 = (1.0 - om - OR0) + drho_l_of_a(1.0, a1, a2)

        def rhs(a, y):
            rc, rl = y
            q = float(q_over_h_of_a(a, a1, a2))
            return [-(3.0 * rc - q) / a, -q / a]

        sol = solve_ivp(rhs, (1.0, 1e-5), [rho_c1, rho_l1], method="DOP853",
                        rtol=1e-12, atol=1e-14, dense_output=True)
        assert sol.success, "la ODE independiente no convergio"
        a_ode = np.geomspace(1e-4, 1.0, 500)
        rc_ode, rl_ode = sol.sol(a_ode)
        err_ode_c = float(np.max(np.abs(rc_ode / fc(a_ode) - 1.0)))
        err_ode_l = float(np.max(np.abs(rl_ode / fl(a_ode) - 1.0)))
        tot_ode = (OR0 * a_ode ** -4 + om * FB * a_ode ** -3
                   + rc_ode + rl_ode)
        err_ode_fr = float(np.max(np.abs(tot_ode
                                         / e2_of_a(a_ode, om, a1, a2) - 1.0)))

        # (d) no hay deficit fosilizado: rho_c/(Oc a^-3) -> 1 para a -> 0
        ratio_early = float(fc(1e-5) / (oc * 1e-5 ** -3))

        if verbose:
            print(f"  a1={a1:+.3f} a2={a2:+.3f}: Friedmann {fr:.1e} | "
                  f"cons {err_c:.1e}/{err_l:.1e} | "
                  f"ODE {err_ode_c:.1e}/{err_ode_l:.1e} "
                  f"(Fried {err_ode_fr:.1e}) | "
                  f"rc/(Oc a^-3)@a=1e-5 = {ratio_early:.8f}")

        assert fr < 1e-13, "Friedmann no cierra"
        assert err_c < 1e-8 and err_l < 1e-8, "conservacion (forma cerrada) falla"
        assert err_ode_c < 1e-6 and err_ode_l < 1e-6, \
            "la ODE no reproduce la forma cerrada"
        assert err_ode_fr < 1e-9, "Friedmann con densidades de la ODE no cierra"
        assert abs(ratio_early - 1.0) < 1e-8, "deficit fosilizado en a->0"

    # --- tests de mutacion: cada uno DEBE fallar ---
    a1, a2 = 0.253, -0.41
    qh_ref = q_over_h_of_a(a_q, a1, a2)

    # M1: delta_rho_c con el factor 2 de mas (error O(1) en la transferencia)
    f1 = lambda x: (2.0 * x / 3.0) * (a1 + 2.0 * a2 * (1.0 - x))     # noqa: E731
    m1 = float(np.max(np.abs(_dlna(f1, a_q) + 3.0 * f1(a_q) - qh_ref)))

    # M2: todo el Delta en el vacio, nada transferido al CDM
    f2 = lambda x: (1.0 - om - OR0) + delta_of_a(x, a1, a2)          # noqa: E731
    m2 = float(np.max(np.abs(_dlna(f2, a_q) + qh_ref)))

    # M3: omite la derivada de Delta (usa -Delta en vez de -(a/3)Delta')
    f3 = lambda x: oc * x ** -3 - delta_of_a(x, a1, a2)              # noqa: E731
    m3 = float(np.max(np.abs(_dlna(f3, a_q) + 3.0 * f3(a_q) - qh_ref)))

    # M4: usa la Q del caso alpha(1-a)^2 con a1=0 (parametro cambiado)
    m4 = float(np.max(np.abs(q_over_h_of_a(a_q, 0.0, a2) - qh_ref)))

    ok = True
    for nm, val in (("delta_rho_c x2", m1), ("todo en el vacio", m2),
                    ("falta dDelta/da", m3), ("Q de otro Delta", m4)):
        good = val > 1e-3
        ok &= good
        if verbose:
            print(f"  mutacion [{nm:16s}]: residuo {val:.2e} "
                  f"{'DETECTADA' if good else 'NO DETECTADA'}")
    assert ok, "test de mutacion ciego"

    # --- w0 exacto y equivalencia con DE desacoplada ---
    OL = 1.0 - om - OR0
    w0_formula = -1.0 + a1 / (3.0 * OL)
    w0_direct = float(w_eff(1.0, om, a1, a2))
    assert abs(w0_formula - w0_direct) < 1e-12, "w0 no coincide"
    eq = (e2_of_a(a_all, om, a1, a2)
          - (OR0 * a_all ** -4 + om * a_all ** -3
             + (OL + delta_of_a(a_all, a1, a2))))
    # tolerancia relativa: a=1e-4 -> E2 ~ 1e16 (redondeo absoluto ~ 1)
    assert np.max(np.abs(eq / e2_of_a(a_all, om, a1, a2))) < 1e-14, \
        "no equivale a DE desacoplada"
    if verbose:
        print(f"  w0: formula {w0_formula:+.6f} == directo {w0_direct:+.6f}")
        print("  equivalencia con DE desacoplada (rho_DE = OL + Delta): OK")
        print("  TODO OK\n")


# ------------------------------------------------------------------
# 3. Ajuste a datos tardios (mismo codigo/starts/tolerancias, regla 6)
# ------------------------------------------------------------------
BOUNDS = {
    "lcdm":   [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0), dvr.RD_BOUNDS],
    "alpha":  [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0), (-2.0, 4.0),
               dvr.RD_BOUNDS],
    "alpha2": [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0), (-4.0, 4.0),
               (-4.0, 4.0), dvr.RD_BOUNDS],
    "w0wa":   [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0), (-3.0, 1.0),
               (-3.0, 2.0), dvr.RD_BOUNDS],
}
NAMES = {
    "lcdm":   ["Om", "H0", "M", "rd"],
    "alpha":  ["Om", "H0", "M", "alpha", "rd"],
    "alpha2": ["Om", "H0", "M", "a1", "a2", "rd"],
    "w0wa":   ["Om", "H0", "M", "w0", "wa", "rd"],
}


def e2_grid(model, p):
    """E2 sobre dvr._ZGRID. Devuelve None si el punto no es fisico."""
    z = dvr._ZGRID
    if model == "lcdm":
        return dvr.E2_lcdm(z, p[0])
    if model == "alpha":
        # OJO: el parametro unico es el coeficiente de (1-a)^2, i.e. a2
        # (el modelo alpha original de la otra IA). Si se pone como a1
        # (coeficiente de (1-a)) el modelo es un w0-CDM con
        # w0 = -1 + a1/(3 OL) y el chi2 mejora por razones ajenas al vacio
        # interactuante (confusion que ya nos costo un mal numero).
        return e2_of_z(z, p[0], 0.0, p[3])
    if model == "alpha2":
        Om, a1, a2 = p[0], p[3], p[4]
        a_chk = np.geomspace(1e-4, 1.0, 400)
        if np.min(rho_c_tilde(a_chk, Om, a1, a2)) <= 0.0:
            return None
        if np.min(rho_l_tilde(a_chk, Om, a1, a2)) <= 0.0:
            return None
        return e2_of_z(z, Om, a1, a2)
    if model == "w0wa":
        if 1.0 - p[0] - OR0 <= 0:
            return None
        return e2_w0wa(z, p[0], p[3], p[4])
    raise ValueError(model)


def e2_w0wa(z, Om, w0, wa):
    """E2 de w0waCDM (CPL), identico al de w0wa_control.py."""
    OL = 1.0 - Om - OR0
    opz = 1.0 + np.asarray(z, dtype=float)
    de = OL * opz ** (3.0 * (1.0 + w0 + wa)) * np.exp(
        -3.0 * wa * (opz - 1.0) / opz)
    return OR0 * opz ** 4 + Om * opz ** 3 + de


def e2_point(model, p, z):
    """E2 en un z escalar/vector, sin volver a comprobar la fisica."""
    if model == "lcdm":
        return dvr.E2_lcdm(z, p[0])
    if model == "alpha":
        # parametro unico = a2 (coef. de (1-a)^2); ver nota en e2_grid.
        return e2_of_z(z, p[0], 0.0, p[3])
    if model == "alpha2":
        return e2_of_z(z, p[0], p[3], p[4])
    if model == "w0wa":
        return e2_w0wa(z, p[0], p[3], p[4])
    raise ValueError(model)


def chi2_total(p, model, sne):
    bnd = BOUNDS[model]
    if len(p) != len(bnd):
        return 1e10
    for v, (lo, hi) in zip(p, bnd):
        if not (lo < v < hi):
            return 1e10
    e2g = e2_grid(model, p)
    if e2g is None or np.any(~np.isfinite(e2g)) or np.any(e2g <= 0):
        return 1e10
    cum = dvr._cumulative_integral(e2g)
    Om, H0, M, rd = p[0], p[1], p[2], p[-1]
    e2f = lambda zz: e2_point(model, p, zz)                         # noqa: E731

    # CC
    c2 = 0.0
    for z, H_obs, sigma in dvr.cc_data:
        e2 = float(np.asarray(e2f(z)))
        if not np.isfinite(e2) or e2 <= 0:
            return 1e10
        c2 += ((H0 * np.sqrt(e2) - H_obs) / sigma) ** 2

    cumf = lambda zz: np.interp(np.asarray(zz, dtype=float),  # noqa: E731
                                dvr._ZGRID, cum)

    # BAO
    for tracer, z, obs_list, corr in dvr.desi_data:
        z = float(z)
        chi_z = float(cumf(z))
        e2 = float(np.asarray(e2f(z)))
        if not np.isfinite(e2) or e2 <= 0:
            return 1e10
        d_m = dvr.c_speed / H0 * chi_z
        d_h = dvr.c_speed / (H0 * np.sqrt(e2))
        if len(obs_list) == 2:
            th = np.array([d_m / rd, d_h / rd])
            obs = np.array([obs_list[0][1], obs_list[1][1]])
            s_m, s_h = obs_list[0][2], obs_list[1][2]
            det = s_m ** 2 * s_h ** 2 * (1.0 - corr ** 2)
            cinv = np.array([[s_h ** 2, -corr * s_m * s_h],
                             [-corr * s_m * s_h, s_m ** 2]]) / det
            d = th - obs
            c2 += float(d @ cinv @ d)
        else:
            d_v = (z * d_m ** 2 * d_h) ** (1.0 / 3.0)
            _, val, sigma = obs_list[0]
            c2 += ((d_v / rd - val) / sigma) ** 2

    # SNe
    z_hd, z_hel, mb, dmb, cov, cinv = sne
    integ = cumf(z_hd)
    dl = (1.0 + z_hel) * dvr.c_speed / H0 * integ
    if np.any(dl <= 0):
        return 1e10
    mu = 5.0 * np.log10(dl) + 25.0 + M
    r = mb - mu
    c2 += float(r @ cinv @ r)
    return c2 if np.isfinite(c2) else 1e10


def starts_for(model):
    """Puntos de partida comunes (mismos para todos los modelos)."""
    base = [[0.30, 70.0, -19.3, 0.0, 147.0],
            [0.30, 68.0, -19.3, 0.6, 147.0],
            [0.30, 70.0, -19.3, -0.05, 147.0],
            [0.25, 73.0, -19.5, 0.0, 155.0],
            [0.35, 67.0, -19.2, 0.1, 140.0],
            [0.32, 68.0, -19.25, -0.02, 147.5]]
    out = []
    for Om, H0, M, x, rd in base:
        if model == "lcdm":
            out.append([Om, H0, M, rd])
        elif model == "alpha":
            out.append([Om, H0, M, x, rd])
        elif model == "alpha2":
            out.append([Om, H0, M, x, 0.0, rd])
            out.append([Om, H0, M, x, -x, rd])       # regimen tipo DESI
        else:
            w0 = -1.0 + x / (3.0 * 0.69)
            out.append([Om, H0, M, w0, -0.4, rd])
    return out


def fit_model(model, sne, maxfev=6000, n_starts=None):
    starts = starts_for(model)
    if n_starts is not None:
        starts = starts[:n_starts]
    best = None
    for x0 in starts:
        res = minimize(chi2_total, np.array(x0, dtype=float),
                       args=(model, sne), method="Nelder-Mead",
                       options=dict(maxiter=maxfev, maxfev=maxfev,
                                    xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    return best


def cpl_equivalent(Om, a1, a2):
    """(w0,wa) CPL que mejor reproduce w_eff(a) en a in [0.4, 1]."""
    a = np.linspace(0.4, 1.0, 300)
    x = 1.0 - a
    w = w_eff(a, Om, a1, a2)
    A = np.column_stack([np.ones_like(x), x])
    (w0, wa), *_ = np.linalg.lstsq(A, w, rcond=None)
    return float(w0), float(wa)


# ------------------------------------------------------------------
# 4. Crecimiento: sistema CDM+bariones en gauge sincrono
# ------------------------------------------------------------------
def growth_two_fluid(Om, a1, a2, N_end=np.log(A_START)):
    """Integra (N = ln a), estado (dc, db, hp) con hp = dh/dN:

        d dc/dN = -1/2 dhp/dN - q(N) dc
        d db/dN = -1/2 dhp/dN
        d hp/dN = -(2 + dlnH/dN) hp - 3 (rc dc + rb db)/E2

    La ultima es la ecuacion de Einstein en gauge sincrono (CDM en reposo),
    que es una RESTRICCION entre (hp, dc, db): integrarla como derivada de
    hpp genera un modo espurio (lo versiones anteriores usaban 4 variables y
    daban +18% la ecuacion estandar). Con a1=a2=0 el sistema se reduce
    exactamente a la ecuacion de crecimiento estandar (ver test
    _test_growth_ic.py).
    """
    def rhs(N, y):
        a = np.exp(N)
        dc, db, hp = y
        e2 = float(e2_of_a(a, Om, a1, a2))
        rc = float(rho_c_tilde(a, Om, a1, a2))
        rb = Om * FB * a ** -3
        q = float(q_over_h_of_a(a, a1, a2)) / rc
        dlnh = 0.5 * float((-4.0 * OR0 * a ** -4 - 3.0 * Om * a ** -3
                            + a * float(ddelta_da(a, a1, a2))) / e2)
        return [-0.5 * hp - q * dc,
                -0.5 * hp,
                -(2.0 + dlnh) * hp - 3.0 * (rc * dc + rb * db) / e2]

    # Condiciones iniciales en el MODO CRECIENTE (era de materia):
    #   delta_c' = -hp/2 con delta ~ a  ->  hp = -2 delta, hpp = -2 delta.
    eps = 1e-3
    y0 = [eps, eps, -2.0 * eps]
    sol = solve_ivp(rhs, (N_end, 0.0), y0, method="DOP853",
                    rtol=1e-10, atol=1e-14, dense_output=True)
    assert sol.success, "la integracion del crecimiento fallo"
    return sol, (lambda N: sol.sol(N)[0]), (lambda N: sol.sol(N)[1])


def growth_standard(Om):
    """Ecuacion de crecimiento estandar (1 fluido) en N, con Delta=0."""
    def rhs(N, y):
        d, dp = y
        a = np.exp(N)
        e2 = float(dvr.E2_lcdm(1.0 / a - 1.0, Om))
        dlnh = 0.5 * (-4.0 * OR0 * a ** -4 - 3.0 * Om * a ** -3) / e2
        om_a = Om * a ** -3 / e2
        return [dp, -(2.0 + dlnh) * dp + 1.5 * om_a * d]

    sol = solve_ivp(rhs, (np.log(A_START), 0.0), [1e-3, 1e-3],
                    method="DOP853", rtol=1e-10, atol=1e-14,
                    dense_output=True)
    assert sol.success
    return lambda N: sol.sol(N)[0]


def growth_diagnostics(Om, a1, a2):
    """D(a) normalizado en A_START, y comparacion con LCDM."""
    sol, Dc, Db = growth_two_fluid(Om, a1, a2)
    _, Dc0, _ = growth_two_fluid(Om, 0.0, 0.0)
    Dstd = growth_standard(Om)
    Nn = np.log(A_START)
    out = {}
    for a in (0.2, 0.5, 0.8, 1.0):
        N = np.log(a)
        g = (Dc(N) / Dc(Nn)) / (Dc0(N) / Dc0(Nn))
        out[a] = g - 1.0
    # consistencia del sistema 2-fluidos con la ecuacion estandar a1=a2=0
    N = 0.0
    f_num = Dc0(N) / Dc0(Nn)
    f_std = Dstd(N) / Dstd(Nn)
    return out, (f_num / f_std - 1.0)


# ------------------------------------------------------------------
def main():
    t0 = time.time()
    args = sys.argv[1:]
    if "--verify-only" in args:
        verify()
        return
    maxfev = 6000
    if "--maxfev" in args:
        maxfev = int(args[args.index("--maxfev") + 1])
    n_starts = None
    if "--starts" in args:
        n_starts = int(args[args.index("--starts") + 1])

    verify()

    print("Cargando datos reales (Pantheon+ / DESI DR1 / CC)...")
    sne = dvr.load_pantheon()
    print(f"  {len(sne[0])} SNe, "
          f"{sum(len(d[2]) for d in dvr.desi_data)} BAO, "
          f"{len(dvr.cc_data)} CC\n")

    models = ["lcdm", "alpha", "alpha2", "w0wa"]
    results = {}
    for model in models:
        print(f"=== {model} ({len(BOUNDS[model])} params) ===")
        t = time.time()
        bf = fit_model(model, sne, maxfev=maxfev, n_starts=n_starts)
        print(f"  chi2_min = {bf.fun:.2f}   ({time.time() - t:.1f} s)")
        print("  " + "  ".join(f"{n}={v:.4f}"
                               for n, v in zip(NAMES[model], bf.x)))
        results[model] = bf

    chi2_l = results["lcdm"].fun
    n_data = (len(sne[0]) + sum(len(d[2]) for d in dvr.desi_data)
              + len(dvr.cc_data))
    print("\n" + "=" * 92)
    print(f"{'modelo':8s} {'k':>2s} {'chi2_min':>10s} {'dchi2':>8s} "
          f"{'dAIC':>7s} {'dBIC':>7s}   parametros")
    print("-" * 92)
    for model in models:
        bf = results[model]
        k = len(bf.x)
        d = bf.fun - chi2_l
        print(f"{model:8s} {k:2d} {bf.fun:10.2f} {d:+8.2f} "
              f"{d + 2 * (k - 4):+7.2f} {d + np.log(n_data) * (k - 4):+7.2f}   "
              f"{'  '.join(f'{n}={v:.4f}' for n, v in zip(NAMES[model], bf.x))}")
    print("=" * 92)

    if "alpha2" in results:
        bf = results["alpha2"]
        Om, a1, a2 = bf.x[0], bf.x[3], bf.x[4]
        OL = 1.0 - Om - OR0
        w0, wa = cpl_equivalent(Om, a1, a2)
        a_chk = np.geomspace(1e-5, 1.0, 4000)
        print("\n--- Modelo a1/a2 en el mejor ajuste tardio ---")
        print(f"  Om={Om:.4f} a1={a1:+.4f} a2={a2:+.4f} rd={bf.x[-1]:.2f}")
        print(f"  OL={OL:.4f}  w0(exacto)={-1 + a1 / (3 * OL):+.4f}  "
              f"CPL equivalente: w0={w0:+.3f} wa={wa:+.3f}")
        print(f"  Delta(0)={a1 + a2:+.4f}  rho_L(a->0)/OL="
              f"{(OL + a1 + a2) / OL:.4f}  materia hoy={Om + a1 / 3.0:.4f}")
        print(f"  min rho_c={np.min(rho_c_tilde(a_chk, Om, a1, a2)):.4g}  "
              f"min rho_L={np.min(rho_l_tilde(a_chk, Om, a1, a2)):.4g}")
        dev = float(np.max(np.abs(
            rho_c_tilde(a_chk, Om, a1, a2) / (Om * (1 - FB) * a_chk ** -3)
            - 1.0)))
        print(f"  max |rho_c/(Oc a^-3) - 1| = {100 * dev:.2f}%")

        print("\n--- Crecimiento: la unica senal que no es DE desacoplada ---")
        gd, consist = growth_diagnostics(Om, a1, a2)
        print(f"  (consistencia 2-fluidos vs ecuacion estandar, a1=a2=0: "
              f"{consist:+.2e})")
        for a, g in gd.items():
            print(f"  a={a:.2f} (z={1/a - 1:.2f}): "
                  f"D_model/D_LCDM - 1 = {100 * g:+.3f}%")
        if "w0wa" in results:
            Omw, w0m, wam = results["w0wa"].x[0], results["w0wa"].x[3], \
                results["w0wa"].x[4]
            print(f"  (referencia w0wa del ajuste: Om={Omw:.4f} "
                  f"w0={w0m:+.4f} wa={wam:+.4f})")

    out = pd.DataFrame([{
        "model": m,
        "n_params": len(results[m].x),
        "n_data": n_data,
        "chi2_min": round(results[m].fun, 2),
        "dchi2_vs_lcdm": round(results[m].fun - chi2_l, 2),
        **{nm: float(v) for nm, v in zip(NAMES[m], results[m].x)},
    } for m in models])
    path = os.path.join(_HERE, "dvch_alpha2_fit.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")
    print(f"Tiempo total: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
