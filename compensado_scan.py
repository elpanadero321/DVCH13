#!/usr/bin/env python3
"""Escaneo sistematico del diseno COMPENSADO: excursiones de materia

    e(a) = A * a^m * (1-a)^p

que van a cero en a=0 y a=1 (materia estandar al principio Y hoy; la
desviacion vive solo a z intermedio). Es la version cuantitativa de la
"interaccion compensada" del plan de fases: Q cambia de signo dentro de
la relacion (Q>0 donde e crece, Q<0 donde e decrece).

Construccion con conservacion EXACTA (unidades rho_crit0, N = ln a):

    rho_c(a) = Oc a^-3 (1+e)          (CDM; bariones Ob a^-3 intactos)
    Q/(H rho_crit0) = d rho_c/dlna + 3 rho_c = Oc a^-2 e'(a)  [identidad]
    rho_L(a) = OL + int_a^1 Q dlnx    (d rho_L/dlna = -Q, rho_L(1) = OL)
    E2 = Or a^-4 + rho_b + rho_c + rho_L

Teorema del area:  q = Q/(H rho_c) = a e'/(1+e)  =>
    int q dlna = ln(1+e(1)) - ln(1+e(0)) = 0
La amortiguacion NETA del crecimiento es exactamente cero para cualquier
excursion compensada. Lo que queda son efectos de FORMA en f*sigma8 y en
H(z). La amplitud del CMB esta a salvo automaticamente
(e(a_rec) ~ A a_rec^m -> 0 para m >= 3).

Dos bugs del compensado_scan.py original, corregidos aqui (eran
regresiones silenciosas): (1) Q~ se calculaba con de/dlna en vez de
de/da (factor a de mas), (2) dlnH/dlna tenia un a^-3 duplicado en el
termino de materia. Ambos se detectan con la suite de verificacion de
abajo (crossover con la familia alfa verificada + test de mutacion).

Etapas:
  verify   : conservacion, limite LCDM, teorema del area, consistencia
             del crecimiento vs alpha_general_fit (familia alfa = caso
             m=4, p=1 con A = 2 a2 / (3 Oc)), mutaciones.
  etapa 1  : puertas baratas para cada (A, m, p): chi2_fsigma8,
             |dH/H|(z=1), max|e|, max|q|, int q dlna, w_eff(0.5),
             e(a_rec), viabilidad (rho_L > 0, 1+e > 0).
  etapa 2  : ajuste tardio (Om, H0, M, rd) de los supervivientes +
             control LCDM (A=0) + control alfa-equivalente (debe
             reproducir chi2 = 1415.10 de alpha_general_fit).

Uso:
    python compensado_scan.py --etapa verify
    python compensado_scan.py --etapa 1
    python compensado_scan.py --etapa 2
    python compensado_scan.py --etapa all
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dvch_vs_lcdm_real as dvr  # noqa: E402

OR0 = dvr.Omega_r0
FB = 0.156                # fraccion barionica (igual que alpha_general_fit)
A_START = 0.02            # inicio de la integracion de crecimiento (z=49)
RSD_Z = np.array([0.02, 0.15, 0.32, 0.57, 0.77, 1.23])
RSD_FS8 = np.array([0.428, 0.45, 0.44, 0.42, 0.39, 0.35])
RSD_ERR = np.array([0.046, 0.04, 0.04, 0.04, 0.04, 0.05])
SIG8_0 = 0.81
N_A = 4000
ZGRID = dvr._ZGRID        # 0..2.61, paso 0.01 (el mismo de la maquinaria)
A_REC = 1.0 / 1101.0


# ------------------------------------------------------------------
# Modelo compensado
# ------------------------------------------------------------------
class CompModel:
    def __init__(self, Om, A, m, p, bug_q=False):
        self.Om, self.Oc, self.Ob = Om, Om * (1.0 - FB), Om * FB
        self.OL = 1.0 - Om - OR0
        self.A, self.m, self.p = A, m, p
        a = np.geomspace(1e-5, 1.0, N_A)
        self.a, self.lna = a, np.log(a)
        self.e = A * a ** m * (1.0 - a) ** p
        # e'(a) analitica: A a^(m-1) (1-a)^(p-1) [m(1-a) - p a]
        ep = A * a ** (m - 1) * (1.0 - a) ** (p - 1) * (m * (1.0 - a) - p * a)
        self.ep = ep
        # Q~ = Oc a^-2 e'  (bug_q = la regresion original: de/dlna sin el a)
        if bug_q:
            self.Q = self.Oc * a ** -3 * (a * ep)
        else:
            self.Q = self.Oc * a ** -2 * ep
        self.rho_c = self.Oc * a ** -3 * (1.0 + self.e)
        self.rho_b = self.Ob * a ** -3
        # rho_L(a) = OL + int_a^1 Q dlnx
        cum = cumulative_trapezoid(self.Q, self.lna, initial=0.0)
        J = cum[-1] - cum                     # int_{a_j}^1 Q dlnx
        self.rho_L = self.OL + J
        self.e2 = OR0 * a ** -4 + self.rho_b + self.rho_c + self.rho_L

    # --- funciones analiticas escalares (crecimiento) ---
    def e_of(self, x):
        return self.A * x ** self.m * (1.0 - x) ** self.p

    def ep_of(self, x):
        return (self.A * x ** (self.m - 1) * (1.0 - x) ** (self.p - 1)
                * (self.m * (1.0 - x) - self.p * x))

    def q_of(self, x):
        return x * self.ep_of(x) / (1.0 + self.e_of(x))

    # --- interpolaciones del fondo ---
    def e2_of_a(self, a):
        return np.interp(a, self.a, self.e2)

    def e2_of_z(self, z):
        a = 1.0 / (1.0 + np.asarray(z, dtype=float))
        return self.e2_of_a(a)

    def late_functions(self):
        """(E2(z), cum dz/E) sobre ZGRID."""
        e2_z = self.e2_of_z(ZGRID)
        inv = 1.0 / np.sqrt(e2_z)
        cum = np.concatenate([[0.0], np.cumsum(
            (inv[1:] + inv[:-1]) * 0.5 * np.diff(ZGRID))])
        return e2_z, cum

    # --- crecimiento: CDM + bariones, gauge sincrono (misma forma que
    #     alpha_general_fit.growth_two_fluid) ---
    def growth(self, bug_dlnh=False):
        def rhs(N, y):
            a = np.exp(N)
            dc, db, hp = y
            e2 = float(self.e2_of_a(a))
            rc = self.Oc * a ** -3 * (1.0 + self.e_of(a))
            rb = self.Ob * a ** -3
            q = float(self.q_of(a))
            if bug_dlnh:
                # la regresion original: a^-3 duplicado en el termino de materia
                dlnh = 0.5 * (-4.0 * OR0 * a ** -4
                              - 3.0 * (rc + rb) * a ** -3) / e2
            else:
                dlnh = 0.5 * (-4.0 * OR0 * a ** -4 - 3.0 * (rc + rb)) / e2
            return [-0.5 * hp - q * dc,
                    -0.5 * hp,
                    -(2.0 + dlnh) * hp - 3.0 * (rc * dc + rb * db) / e2]

        eps = 1e-3
        sol = solve_ivp(rhs, (np.log(A_START), 0.0), [eps, eps, -2.0 * eps],
                        method="DOP853", rtol=1e-10, atol=1e-14,
                        dense_output=True)
        assert sol.success, "la integracion del crecimiento fallo"
        return sol

    def chi2_fs8(self, sigma8_0=SIG8_0, bug_dlnh=False):
        sol = self.growth(bug_dlnh=bug_dlnh)
        D = lambda N: sol.sol(N)[0]                       # noqa: E731
        Ns = np.log(A_START)
        fs = []
        for z in RSD_Z:
            N = np.log(1.0 / (1.0 + z))
            h = 1e-4
            f = (D(N + h) - D(N - h)) / (2 * h * D(N))
            fs.append(f * sigma8_0 * D(N) / D(Ns))
        fs = np.array(fs)
        return float(np.sum(((fs - RSD_FS8) / RSD_ERR) ** 2)), fs

    def dH_over_H(self, z):
        return (np.sqrt(self.e2_of_z(z)) / np.sqrt(dvr.E2_lcdm(z, self.Om))
                - 1.0)

    def w_eff(self, z):
        a = 1.0 / (1.0 + z)
        dl = np.gradient(self.rho_L, self.lna)   # d rho_L/dlna
        w = -1.0 - dl / (3.0 * self.rho_L)
        return float(np.interp(a, self.a, w))

    def area_integral(self, a0=0.02):
        """int q dlna de a0 a 1 (teorema: = ln((1+e(1))/(1+e(a0))) ~ 0)."""
        aq = np.linspace(a0, 1.0, 4001)
        q = self.q_of(aq)
        return float(np.trapezoid(q, np.log(aq)))


# ------------------------------------------------------------------
# Verificacion
# ------------------------------------------------------------------
def alpha_equiv_params(Om, a2):
    """La familia alfa pura (a1=0, Delta = a2(1-a)^2) en coords (A,m,p):
    e = (2 a2 /(3 Oc)) a^4 (1-a)  ->  A = 2 a2/(3 Oc), m=4, p=1."""
    Oc = Om * (1.0 - FB)
    return 2.0 * a2 / (3.0 * Oc), 4, 1


def verify():
    import alpha_general_fit as ag

    print("=" * 78)
    print("VERIFICACION del modelo compensado (crossover con la familia alfa)")
    print("=" * 78)
    Om, a2 = 0.29619247139654903, 0.5970453251977522   # mejor fit alpha
    A, m, p = alpha_equiv_params(Om, a2)

    # (1) limite LCDM: A=0 reproduce E2_lcdm exacto
    mm0 = CompModel(Om, 0.0, 4, 2)
    rel = float(np.max(np.abs(
        mm0.e2 / dvr.E2_lcdm(1.0 / mm0.a - 1.0, Om) - 1.0)))
    print(f"  (1) limite A=0 vs E2_lcdm: max rel {rel:.2e}")
    assert rel < 1e-12, "el limite LCDM no es exacto"

    # (2) Friedmann hoy: E2(a=1) = 1
    mm = CompModel(Om, A, m, p)
    print(f"  (2) E2(a=1) - 1 = {mm.e2[-1] - 1.0:.2e}")
    assert abs(mm.e2[-1] - 1.0) < 1e-9, "Friedmann hoy no cierra"

    # (3) conservacion: d rho_c/dlna + 3 rho_c = Q~ (numerico vs regla)
    drc = np.gradient(mm.rho_c, mm.lna)
    res_c = float(np.max(np.abs((drc + 3.0 * mm.rho_c - mm.Q)
                                 / mm.rho_c)))
    drl = np.gradient(mm.rho_L, mm.lna)
    res_l = float(np.max(np.abs((drl + mm.Q) / mm.OL)))
    print(f"  (3) conservacion: residuos {res_c:.2e} (CDM) / "
          f"{res_l:.2e} (vacio)")
    assert res_c < 1e-6 and res_l < 1e-6, "conservacion falla"

    # (4) crossover con la familia alfa verificada: mismos E2, Q y D(z)
    a_chk = np.geomspace(1e-4, 1.0, 800)
    dE2 = float(np.max(np.abs(
        mm.e2_of_a(a_chk) / ag.e2_of_a(a_chk, Om, 0.0, a2) - 1.0)))
    Q_rule = ag.q_over_h_of_a(a_chk, 0.0, a2)
    dQ = float(np.max(np.abs(mm.Q / np.interp(a_chk, mm.a, mm.Q)
                             - 1.0)))
    Q_at = np.interp(a_chk, mm.a, mm.Q)
    dQ = float(np.max(np.abs(Q_at - Q_rule) / np.maximum(Q_rule, 1e-12)))
    print(f"  (4) crossover alfa: max rel E2 {dE2:.2e}, max rel Q {dQ:.2e}")
    assert dE2 < 1e-6 and dQ < 1e-6, "no coincide con la familia alfa"

    # (5) teorema del area: int q dlna = ln((1+e(1))/(1+e(a0)))
    for (A_, m_, p_) in [(1.5, 4, 2), (0.7, 6, 3), (2.0, 3, 1)]:
        mmx = CompModel(Om, A_, m_, p_)
        num = mmx.area_integral()
        ana = np.log((1.0 + mmx.e_of(1.0)) / (1.0 + mmx.e_of(0.02)))
        print(f"  (5) area A={A_} m={m_} p={p_}: int q dlna = {num:+.2e} "
              f"(analitico {ana:+.2e})")
        assert abs(num - ana) < 1e-6, "el teorema del area falla"

    # (6) crecimiento: A=0 coincide con la ecuacion estandar
    sol0 = mm0.growth()
    D0 = lambda N: sol0.sol(N)[0]                       # noqa: E731
    Dstd = ag.growth_standard(Om)
    Ns = np.log(A_START)
    worst = 0.0
    for z in (0.0, 0.5, 1.0, 2.0):
        N = np.log(1.0 / (1.0 + z))
        worst = max(worst, abs(D0(N) / D0(Ns) / (Dstd(N) / Dstd(Ns)) - 1.0))
    print(f"  (6) crecimiento A=0 vs ecuacion estandar: max rel {worst:.2e}")
    assert worst < 1e-4, "el sistema de crecimiento no reduce a LCDM"

    # (7) crecimiento: coincide con growth_two_fluid de la familia alfa
    sol = mm.growth()
    Dc = lambda N: sol.sol(N)[0]                        # noqa: E731
    _, Da, _ = ag.growth_two_fluid(Om, 0.0, a2)
    worst = 0.0
    for z in (0.0, 0.25, 0.5, 1.0):
        N = np.log(1.0 / (1.0 + z))
        worst = max(worst, abs(Dc(N) / Dc(Ns) / (Da(N) / Da(Ns)) - 1.0))
    print(f"  (7) crecimiento vs alpha growth_two_fluid: max rel {worst:.2e}")
    assert worst < 1e-4, "el crecimiento no coincide con la familia alfa"

    # (8) mutaciones: cada una DEBE detectarse
    c2_ok, _ = mm.chi2_fs8()
    c2_qbug, _ = mm.chi2_fs8()  # placeholder, se reemplaza abajo
    mmb = CompModel(Om, A, m, p, bug_q=True)
    c2_qbug, _ = mmb.chi2_fs8()
    c2_dlnh, _ = mm.chi2_fs8(bug_dlnh=True)
    print(f"  (8) chi2_fs8 correcto={c2_ok:.1f} | mutacion Q({c2_qbug:.1f}) "
          f"debe diferir | mutacion dlnh({c2_dlnh:.1f}) debe diferir")
    assert abs(c2_qbug - c2_ok) > 1.0, "mutacion Q no detectada"
    assert abs(c2_dlnh - c2_ok) > 1.0, "mutacion dlnh no detectada"
    # mutacion rho_L con signo cambiado: rompe Friedmann hoy
    mmw = CompModel(Om, A, m, p)
    cum = cumulative_trapezoid(mmw.Q, mmw.lna, initial=0.0)
    J = cum[-1] - cum
    e2_wrong = (OR0 * mmw.a ** -4 + mmw.rho_b + mmw.rho_c
                + (mmw.OL - J))
    print(f"  (8b) mutacion rho_L(-J): |E2(1)-1| = {abs(e2_wrong[-1]-1.0):.2e}"
          " (debe ser >> 0)")
    assert abs(e2_wrong[-1] - 1.0) > 1e-3, "mutacion rho_L no detectada"

    print("  TODO OK\n")


# ------------------------------------------------------------------
# Etapa 1: puertas baratas
# ------------------------------------------------------------------
GRID_A = (0.15, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0,
          -0.15, -0.3, -0.5, -0.7, -1.0)
GRID_M = (3, 4, 6, 8)
GRID_P = (1, 2, 3, 4)


def stage1(quick=False):
    Om = 0.315
    A_list = (0.3, 1.0, -0.3) if quick else GRID_A
    m_list = (4,) if quick else GRID_M
    p_list = (2,) if quick else GRID_P
    rows = []
    t0 = time.time()
    print(f"Etapa 1: {len(A_list)*len(m_list)*len(p_list)} modelos "
          f"(puertas baratas)")
    print(f"{'A':>5} {'m':>3} {'p':>3} {'chi2fs8':>8} {'dH/H@1':>9} "
          f"{'dH/H@0.5':>9} {'max|e|':>8} {'max|q|':>7} {'area':>8} "
          f"{'w_eff(.5)':>10} {'e_rec':>8} {'minrL':>7} {'viable':>6}")
    for A in A_list:
        for m in m_list:
            for p in p_list:
                mm = CompModel(Om, A, m, p)
                minrL = float(mm.rho_L.min())
                mine = float((1.0 + mm.e).min())
                viable = minrL > 0.0 and mine > 0.05
                if not viable:
                    rows.append(dict(A=A, m=m, p=p, viable=False))
                    continue
                c2, _ = mm.chi2_fs8()
                dh1 = float(mm.dH_over_H(1.0))
                dh05 = float(mm.dH_over_H(0.5))
                ae = np.linspace(0.05, 1.0, 2001)
                maxq = float(np.max(np.abs(mm.q_of(ae))))
                maxe = float(np.max(np.abs(mm.e)))
                area = mm.area_integral()
                w5 = mm.w_eff(0.5)
                e_rec = float(mm.e_of(A_REC))
                rows.append(dict(A=A, m=m, p=p, viable=True, chi2_fs8=c2,
                                 dH1=dh1, dH05=dh05, max_e=maxe, max_q=maxq,
                                 area=area, w_eff=w5, e_rec=e_rec,
                                 min_rho_L=minrL, min_1pe=mine))
                print(f"{A:5.2f} {m:3d} {p:3d} {c2:8.1f} {100*dh1:+8.2f}% "
                      f"{100*dh05:+8.2f}% {100*maxe:7.2f}% {maxq:7.3f} "
                      f"{area:+8.1e} {w5:+10.3f} {e_rec:8.1e} {minrL:7.4f} "
                      f"{'SI' if c2 < 12 else 'no':>6}")
    import pandas as pd
    df = pd.DataFrame(rows)
    path = os.path.join(_HERE, "compensado_stage1.csv")
    df.to_csv(path, index=False)
    nv = df[df.get("viable", False) == True]                      # noqa: E712
    print(f"\nEtapa 1: {len(nv)}/{len(df)} viables, "
          f"{int((nv.get('chi2_fs8', pd.Series(dtype=float)) < 12).sum())} "
          f"pasan chi2_fs8<12. Tiempo {time.time()-t0:.0f}s. Wrote {path}")
    return df


# ------------------------------------------------------------------
# Etapa 2: ajuste tardio de los supervivientes
# ------------------------------------------------------------------
def chi2_late(params, A, m, p, sne):
    Om, H0, M, rd = params
    if not (0.10 < Om < 0.50 and 60.0 < H0 < 80.0 and -25.0 < M < -18.0
            and 130.0 < rd < 180.0):
        return 1e10
    mm = CompModel(Om, A, m, p)
    if float(mm.rho_L.min()) <= 0.0:
        return 1e10
    e2_z, cum = mm.late_functions()
    e2f = lambda z: np.interp(z, ZGRID, e2_z)                     # noqa: E731
    cumf = lambda z: np.interp(np.asarray(z, dtype=float),        # noqa: E731
                               ZGRID, cum)
    c2 = 0.0
    for z, H_obs, sigma in dvr.cc_data:
        e2 = float(e2f(z))
        if e2 <= 0:
            return 1e10
        c2 += ((H0 * np.sqrt(e2) - H_obs) / sigma) ** 2
    for tracer, z, obs_list, corr in dvr.desi_data:
        z = float(z)
        chi_z = float(cumf(z))
        e2 = float(e2f(z))
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
    z_hd, z_hel, mb, dmb, cov, cinv = sne
    dl = (1.0 + z_hel) * dvr.c_speed / H0 * cumf(z_hd)
    if np.any(dl <= 0):
        return 1e10
    mu = 5.0 * np.log10(dl) + 25.0 + M
    r = mb - mu
    c2 += float(r @ cinv @ r)
    return c2 if np.isfinite(c2) else 1e10


def fit_late(A, m, p, sne, n_starts=4, label=""):
    starts = [[0.30, 70.0, -19.3, 147.0],
              [0.25, 73.0, -19.5, 155.0],
              [0.35, 67.0, -19.2, 140.0],
              [0.28, 71.0, -19.4, 165.0]]
    best = None
    for x0 in starts[:n_starts]:
        res = minimize(chi2_late, np.array(x0, dtype=float),
                       args=(A, m, p, sne), method="Nelder-Mead",
                       options=dict(maxiter=4000, maxfev=4000,
                                    xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    print(f"  {label or f'A={A} m={m} p={p}'}: chi2 = {best.fun:.2f} "
          f"(Om={best.x[0]:.4f} H0={best.x[1]:.2f} M={best.x[2]:.3f} "
          f"rd={best.x[3]:.1f})")
    return best


def stage2(sne, quick=False):
    import pandas as pd
    path = os.path.join(_HERE, "compensado_stage1.csv")
    df = pd.read_csv(path)
    df = df[df["viable"] == True]                                 # noqa: E712
    df = df[df["chi2_fs8"] < 12.0]
    # seleccion: para cada (m,p) el mejor chi2_fs8 y el de maximo |dH/H(z=1)|
    sel = []
    for (m, p), grp in df.groupby(["m", "p"]):
        best = grp.loc[grp["chi2_fs8"].idxmin()]
        sel.append((best["A"], m, p, float(best["chi2_fs8"]),
                    float(best["dH1"])))
        big = grp.loc[grp["dH1"].abs().idxmax()]
        if big["A"] != best["A"]:
            sel.append((big["A"], m, p, float(big["chi2_fs8"]),
                        float(big["dH1"])))
    sel = sorted(set((a, m, p) for a, m, p, *_ in sel))
    if quick:
        sel = sel[:3]

    print("Etapa 2: ajustes tardios (Om,H0,M,rd) de los supervivientes")
    # controles
    print("  --- controles ---")
    t0 = time.time()
    bf_lcdm = fit_late(0.0, 4, 2, sne, label="LCDM (A=0)")
    Om_alpha, a2_alpha = 0.29619247139654903, 0.5970453251977522
    A_al, m_al, p_al = alpha_equiv_params(Om_alpha, a2_alpha)
    bf_alpha = fit_late(A_al, m_al, p_al, sne,
                        label=f"alfa-equiv (A={A_al:.3f},m=4,p=1)")
    print(f"  (referencia alpha_general_fit: LCDM 1416.51, alpha 1415.10)")

    results = []
    for A, m, p in sel:
        t = time.time()
        bf = fit_late(A, m, p, sne, label=f"A={A} m={m} p={p}")
        # fs8 en el mejor fit
        mm = CompModel(bf.x[0], A, m, p)
        c2f, _ = mm.chi2_fs8()
        results.append(dict(A=A, m=m, p=p, chi2_late=bf.fun,
                            dchi2=bf.fun - bf_lcdm.fun,
                            Om=bf.x[0], H0=bf.x[1], M=bf.x[2], rd=bf.x[3],
                            chi2_fs8_at_best=c2f))
        print(f"      -> dchi2={bf.fun-bf_lcdm.fun:+.2f} "
              f"chi2_fs8={c2f:.1f} ({time.time()-t:.0f}s)")

    rows = [dict(model="lcdm(A=0)", A=0, chi2_late=bf_lcdm.fun,
                 dchi2=0.0, chi2_fs8=np.nan,
                 Om=bf_lcdm.x[0], H0=bf_lcdm.x[1], M=bf_lcdm.x[2],
                 rd=bf_lcdm.x[3]),
            dict(model="alpha-equiv", A=A_al, chi2_late=bf_alpha.fun,
                 dchi2=bf_alpha.fun - bf_lcdm.fun, chi2_fs8=np.nan,
                 Om=bf_alpha.x[0], H0=bf_alpha.x[1], M=bf_alpha.x[2],
                 rd=bf_alpha.x[3])]
    for r in results:
        rows.append(dict(model=f"A={r['A']} m={r['m']} p={r['p']}",
                         A=r["A"], chi2_late=r["chi2_late"],
                         dchi2=r["dchi2"],
                         chi2_fs8=r["chi2_fs8_at_best"],
                         Om=r["Om"], H0=r["H0"], M=r["M"], rd=r["rd"]))
    out = pd.DataFrame(rows)
    path2 = os.path.join(_HERE, "compensado_stage2.csv")
    out.to_csv(path2, index=False)
    print("\n" + "=" * 90)
    print(f"{'modelo':20s} {'chi2_late':>10s} {'dchi2':>8s} "
          f"{'chi2_fs8':>9s}")
    for _, r in out.iterrows():
        fs = f"{r['chi2_fs8']:.1f}" if np.isfinite(r["chi2_fs8"]) else "  -"
        print(f"{r['model']:20s} {r['chi2_late']:10.2f} {r['dchi2']:+8.2f} "
              f"{fs:>9s}")
    print("=" * 90)
    print(f"Tiempo etapa 2: {time.time()-t0:.0f}s. Wrote {path2}")


# ------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    etapa = "all"
    for i, a in enumerate(args):
        if a == "--etapa" and i + 1 < len(args):
            etapa = args[i + 1]
    if etapa in ("verify", "all"):
        verify()
    if etapa in ("1", "all"):
        stage1(quick=("--quick" in args))
    if etapa in ("2", "all"):
        print("Cargando Pantheon+...")
        sne = dvr.load_pantheon()
        print(f"  {len(sne[0])} SNe")
        stage2(sne, quick=("--quick" in args))


if __name__ == "__main__":
    main()
