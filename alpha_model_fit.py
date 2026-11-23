"""Hipotesis alternativa (cerradura alpha): verificacion independiente + ajuste
a los datos tardios reales (los mismos de dvch_vs_lcdm_real.py).

Modelo (a = 1/(1+z)):
    E2(a) = Or0*a^-4 + Om0*a^-3 + OL0 + alpha*(1-a)^2
    rho_c(a)   = Oc0*a^-3 + (2*alpha/3)*a*(1-a)      (CDM; bariones a^-3)
    rho_L(a)   = OL0   + (alpha/3)*(1-a)*(3-5*a)

alpha=0 recupera LCDM exactamente. A diferencia del DVCH original, la
desviacion de materia ~ a^4*(1-a) -> 0 en el pasado (no hay deficit fosilizado).
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import dvch_vs_lcdm_real as dvr  # noqa: E402

# ------------------------------------------------------------------
# 1. Verificacion independiente (derivadas analiticas a mano)
# ------------------------------------------------------------------
OR0 = dvr.Omega_r0


def e2_alpha(z, Om, alpha):
    """E2(z) con la cerradura alpha (forma cerrada)."""
    OL = 1.0 - Om - OR0
    opz = 1.0 + np.asarray(z, dtype=float)
    return OR0 * opz**4 + Om * opz**3 + OL + alpha * (1.0 - 1.0 / opz) ** 2


def rho_c_alpha(a, Om, alpha, fb=0.156):
    """Densidad de CDM (bariones a parte, fb = fraccion barionica)."""
    Oc = Om * (1.0 - fb)
    return Oc * a**-3 + (2.0 * alpha / 3.0) * a * (1.0 - a)


def rho_l_alpha(a, Om, alpha):
    OL = 1.0 - Om - OR0
    return OL + (alpha / 3.0) * (1.0 - a) * (3.0 - 5.0 * a)


def q_over_h(a, alpha):
    """Q/(H*rho_crit0) tal como esta documentado en PENDIENTES_ALPHA.md.

    Convencion: d(rho_c)/dt + 3H rho_c = +Q (solo CDM), d(rho_L)/dt = -Q.
    """
    return (2.0 * alpha / 3.0) * a * (4.0 - 5.0 * a)


def _dlna(f, a, h=1e-20):
    """a*df/da de la funcion IMPLEMENTADA, por paso complejo.

    f(a*e^{ih}) = f(a) + i*h*a*f'(a) + O(h^2): no hay resta de numeros
    grandes, asi que la derivada es exacta a precision de maquina.
    """
    return np.imag(f(a * np.exp(1j * h))) / h


def verify():
    """Chequeos propios: Friedmann, conservacion con Q, normalizacion.

    NOTA (bug corregido): la version anterior dividia el residuo de
    conservacion por 3*Oc*(1e-4)^-3 ~ 8e11, de modo que un error O(1) en
    la derivada del vacio seguia pasando el assert (test vacio). Ahora se
    comparan las derivadas de las funciones implementadas contra la Q
    documentada, sin reescalar, y se incluye un test de mutacion que DEBE
    fallar.
    """
    print("=== Verificacion independiente de la cerradura alpha ===")
    om, fb = 0.315, 0.156
    oc, ob = om * (1 - fb), om * fb
    a_all = np.geomspace(1e-4, 1.0, 2001)
    # Para Q basta a >= 1e-2: alli doble precision resuelve Q/H ~ alpha*a
    # frente a Oc*a^-3 (residuo de redondeo ~1e-10 << tolerancia 1e-8).
    a_q = np.geomspace(1e-2, 1.0, 2001)
    g_peak = 0.8**4 * 0.2  # max de a^4 (1-a), en a = 0.8 (z = 0.25)
    for alpha in (-0.1, 0.0, 0.1, 0.3, 1.0):
        fc = lambda x, al=alpha: rho_c_alpha(x, om, al, fb)  # noqa: E731
        fl = lambda x, al=alpha: rho_l_alpha(x, om, al)  # noqa: E731
        total = OR0 * a_all**-4 + ob * a_all**-3 + fc(a_all) + fl(a_all)
        fr = np.max(np.abs(total / e2_alpha(1.0 / a_all - 1.0, om, alpha)
                           - 1.0))
        qh = q_over_h(a_q, alpha)
        tol = 1e-8 * max(1.0, abs(alpha))
        # CDM:   a d(rho_c)/da + 3 rho_c = +Q/H
        err_c = np.max(np.abs(_dlna(fc, a_q) + 3.0 * fc(a_q) - qh))
        # Vacio: a d(rho_L)/da = -Q/H
        err_l = np.max(np.abs(_dlna(fl, a_q) + qh))
        # Normalizacion y desviacion maxima del CDM respecto a Oc*a^-3
        r = fc(a_all) / (oc * a_all**-3)
        dev = np.max(np.abs(r - 1.0))
        dev_th = 2.0 * abs(alpha) / (3.0 * oc) * g_peak
        print(f"  alpha={alpha:+.2f}: |Friedmann-1|={fr:.2e} "
              f"|cons CDM|={err_c:.2e} |cons vac|={err_l:.2e} "
              f"ratio CDM@z=1e4: {r[0]:.6f} @hoy: {r[-1]:.6f} "
              f"max dev: {dev*100:.2f}% (teoria {dev_th*100:.2f}%, z=0.25)")
        assert fr < 1e-14, "Friedmann no cierra"
        assert err_c < tol, "rho_c no cumple rho_c' + 3H rho_c = Q"
        assert err_l < tol, "rho_L no cumple rho_L' = -Q"
        assert abs(r[0] - 1) < 1e-12 and abs(r[-1] - 1) < 1e-12
        assert abs(dev - dev_th) <= 1e-3 * dev_th + 1e-15

    # Test de mutacion: un vacio con amplitud erronea (2*alpha) debe FALLAR.
    alpha = 0.3
    bad = np.max(np.abs(_dlna(lambda x: rho_l_alpha(x, om, 2 * alpha), a_q)
                        + q_over_h(a_q, alpha)))
    assert bad > 1e3 * 1e-8, "el test de conservacion no es sensible"
    print(f"  mutacion (vacio con 2*alpha): residuo={bad:.2e} -> detectado")
    print("  TODO OK: Friedmann, conservacion (Q), normalizacion en ambos "
          "limites.\n")


# ------------------------------------------------------------------
# 2. Ajuste a los datos tardios (misma likelihood que el pipeline)
# ------------------------------------------------------------------
ALPHA_BOUNDS = (-2.0, 4.0)


def model_funcs_alpha(Om, alpha):
    """(E2(z), cum dz/E) sobre el mismo grid que el resto del pipeline."""
    e2_grid = e2_alpha(dvr._ZGRID, Om, alpha)
    if np.any(~np.isfinite(e2_grid)) or np.any(e2_grid <= 0):
        return None
    return (lambda z: e2_alpha(z, Om, alpha),
            lambda z: np.interp(np.asarray(z, dtype=float), dvr._ZGRID,
                                dvr._cumulative_integral(e2_grid)))


def total_chi2_alpha(params, sne):
    Om, H0, M, alpha, rd = params
    if not (0.10 < Om < 0.50 and 60.0 < H0 < 80.0 and -25.0 < M < -18.0
            and ALPHA_BOUNDS[0] < alpha < ALPHA_BOUNDS[1]
            and dvr.RD_BOUNDS[0] < rd < dvr.RD_BOUNDS[1]):
        return 1e10
    funcs = model_funcs_alpha(Om, alpha)
    if funcs is None:
        return 1e10
    e2f, cumf = funcs
    # CC
    c2 = 0.0
    for z, H_obs, sigma in dvr.cc_data:
        e2 = e2f(z)
        c2 += ((H0 * np.sqrt(e2) - H_obs) / sigma) ** 2
    # BAO (igual que chi2_bao)
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
            det = s_m**2 * s_h**2 * (1.0 - corr**2)
            cinv = np.array([[s_h**2, -corr * s_m * s_h],
                             [-corr * s_m * s_h, s_m**2]]) / det
            d = th - obs
            c2 += float(d @ cinv @ d)
        else:
            d_v = (z * d_m**2 * d_h) ** (1.0 / 3.0)
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


def best_fit_alpha(sne, n_restarts=10):
    from scipy.optimize import minimize
    starts = [[0.30, 70.0, -19.3, 0.0, 147.0],
              [0.30, 68.0, -19.3, -0.05, 147.0],
              [0.30, 70.0, -19.3, +0.05, 147.0],
              [0.25, 73.0, -19.5, -0.1, 155.0],
              [0.35, 67.0, -19.2, +0.1, 140.0],
              [0.32, 68.0, -19.25, -0.02, 147.5],
              [0.28, 71.0, -19.4, +0.02, 150.0],
              [0.30, 69.0, -19.35, 0.01, 148.0],
              [0.30, 70.0, -19.3, +1.0, 147.0],
              [0.30, 70.0, -19.3, -1.0, 147.0]]
    best = None
    for x0 in starts:
        res = minimize(total_chi2_alpha, np.array(x0), args=(sne,),
                       method="Nelder-Mead",
                       options=dict(maxiter=8000, maxfev=8000,
                                    xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    return best


def run_mcmc_alpha(sne, n_walkers=32, n_steps=4000, burn=1000, start_from=None):
    """MCMC con diagnostico de convergencia (E4): autocorrelacion y
    fraccion de aceptacion; arranque desde el mejor ajuste."""
    import emcee
    ndim = 5
    p0 = np.array(start_from if start_from is not None
                  else [0.30, 68.3, -19.41, 0.6, 146.8])
    scale = np.array([0.01, 0.3, 0.03, 0.15, 1.5])

    def logpost(p):
        Om, H0, M, alpha, rd = p
        if not (0.10 < Om < 0.50 and 60.0 < H0 < 80.0 and -25.0 < M < -18.0
                and ALPHA_BOUNDS[0] < alpha < ALPHA_BOUNDS[1]
                and dvr.RD_BOUNDS[0] < rd < dvr.RD_BOUNDS[1]):
            return -np.inf
        c2 = total_chi2_alpha(p, sne)
        if c2 > 1e9:
            return -np.inf
        return -0.5 * c2

    start = p0 + scale * np.random.randn(n_walkers, ndim)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, logpost)
    sampler.run_mcmc(start, n_steps, progress=False)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.max(tau))
        print(f"  autocorr time max = {tau_max:.1f} "
              f"(n_steps/burn = {n_steps}/{burn}, ratio "
              f"{n_steps / max(tau_max, 1e-9):.1f})")
        if n_steps <= 50 * tau_max:
            print("  AVISO: n_steps < 50*tau_max (convergencia dudosa)")
    except Exception as exc:
        print(f"  (no se pudo estimar tau: {exc})")
    print(f"  acceptance fraction = {np.mean(sampler.acceptance_fraction):.3f}")
    chain = sampler.get_chain(discard=burn, flat=True)
    return chain


def best_fit_lcdm_inscript(sne):
    """LCDM (alpha=0) minimizado en EXACTAMENTE las mismas condiciones que
    el modelo alpha (E1): mismos starts, tolerancias y maxfev."""
    from scipy.optimize import minimize

    def chi2_fixed0(p):
        Om, H0, M, rd = p
        return total_chi2_alpha([Om, H0, M, 0.0, rd], sne)

    starts = [[0.30, 70.0, -19.3, 147.0],
              [0.25, 73.0, -19.5, 155.0],
              [0.35, 67.0, -19.2, 140.0],
              [0.32, 68.0, -19.25, 147.5],
              [0.28, 71.0, -19.4, 150.0],
              [0.30, 69.0, -19.35, 148.0],
              [0.20, 75.0, -19.4, 165.0],
              [0.30, 68.0, -19.3, 147.0]]
    best = None
    for x0 in starts:
        res = minimize(chi2_fixed0, np.array(x0), method="Nelder-Mead",
                       options=dict(maxiter=8000, maxfev=8000,
                                    xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    return best


def load_chi2_w0wa():
    """Lee chi2_min de w0waCDM del CSV (E2)."""
    path = os.path.join(_HERE, "dvch_w0wa_fit.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Falta {path}. Corre chi2_w0wa.py primero.")
    return float(pd.read_csv(path).chi2_min.iloc[0])


def main():
    verify()
    print("Cargando datos reales (Pantheon+ / DESI DR1 / CC)...")
    sne = dvr.load_pantheon()
    print(f"  {len(sne[0])} SNe, "
          f"{sum(len(d[2]) for d in dvr.desi_data)} BAO, "
          f"{len(dvr.cc_data)} CC\n")

    print("=== MODELO alpha (Om, H0, M, alpha, rd) ===")
    bf = best_fit_alpha(sne)
    Om, H0, M, alpha, rd = bf.x
    print(f"  chi2_min = {bf.fun:.2f}")
    print(f"    Om={Om:.4f}  H0={H0:.3f}  M={M:.4f}  alpha={alpha:.5f}  "
          f"rd={rd:.2f}")

    print("  LCDM anidado (alpha=0, misma config)...")
    bf_lcdm = best_fit_lcdm_inscript(sne)
    print(f"    chi2_min LCDM = {bf_lcdm.fun:.2f}")

    print("  MCMC (emcee)...")
    chain = run_mcmc_alpha(sne, start_from=bf.x)
    names = ["Om", "H0", "M", "alpha", "rd"]
    q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
    for nm, v, lo, hi in zip(names, q50, q50 - q16, q84 - q50):
        print(f"    {nm:6s} = {v:.4f} +{hi:.4f} -{lo:.4f}")
    np.save(os.path.join(_HERE, "dvch_alpha_chain.npy"), chain)

    # Comparacion: LCDM minimizado en las mismas condiciones (E1) y w0wa
    # leido del CSV (E2)
    summary = pd.read_csv(os.path.join(_HERE, "dvch_late_time_fit_summary.csv"))
    chi2_lcdm_csv = float(summary[summary.model == "lcdm"].chi2_min.iloc[0])
    chi2_lcdm = float(bf_lcdm.fun)
    chi2_dvch = float(summary[summary.model == "dvch"].chi2_min.iloc[0])
    chi2_w0wa = load_chi2_w0wa()
    k_lcdm, k_alpha, k_w0wa = 4, 5, 6
    n_data = len(sne[0]) + sum(len(d[2]) for d in dvr.desi_data) + len(dvr.cc_data)
    print("\n=== Comparacion ===")
    print(f"  LCDM (in-script, misma config) : chi2={chi2_lcdm:.2f}"
          f"  (CSV pipeline: {chi2_lcdm_csv:.2f})")
    print(f"  alpha   : chi2={bf.fun:.2f}  (1 param extra)")
    print(f"  w0waCDM : chi2={chi2_w0wa:.2f}  (2 params extra, "
          f"de dvch_w0wa_fit.csv)")
    print(f"  DVCH    : chi2={chi2_dvch:.2f}  (2 params extra)")
    d = bf.fun - chi2_lcdm
    print(f"  Delta chi2 (alpha - LCDM) = {d:+.2f}")
    print(f"  Delta AIC = {d + 2 * 1:+.2f}   Delta BIC = {d + np.log(n_data):+.2f}")

    # Diagnosticos del fondo en el mejor ajuste (E3)
    print("\n=== Diagnostico del fondo (mejor ajuste) ===")
    zs = np.array([0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0])
    e2m = e2_alpha(zs, Om, alpha)
    e2l = dvr.E2_lcdm(zs, Om)
    # tambien contra el mejor ajuste LCDM (no solo el mismo Om)
    e2l_bf = dvr.E2_lcdm(zs, *bf_lcdm.x[:1])
    dh_bf = np.sqrt(e2m * (bf_lcdm.x[1] / H0) ** 2 / e2l_bf) - 1.0
    dh = np.sqrt(e2m / e2l) - 1.0
    print("  dH/H vs LCDM(mismo Om):")
    for z, h in zip(zs, dh):
        print(f"    z={z:>5}: {h:+.2e}")
    print("  H absoluto vs LCDM best-fit (H0_LCDM="
          f"{bf_lcdm.x[1]:.2f}):")
    for z, h in zip(zs, dh_bf):
        print(f"    z={z:>5}: {h:+.2e}")
    a_fine = np.linspace(1e-3, 1.0, 20001)
    fb_eff = 1 - (1 - 0.156)
    r = rho_c_alpha(a_fine, Om, alpha, fb=0.156) / (Om * 0.844 * a_fine**-3)
    dev = float(np.max(np.abs(r - 1.0)))
    z_max = float(1.0 / a_fine[int(np.argmax(np.abs(r - 1.0)))] - 1.0)
    print(f"  max |ratio CDM/(Oc a^-3) - 1| = {dev * 100:.2f}% "
          f"(en z = {z_max:.3f}); deficit fosilizado = 0")

    out = pd.DataFrame([{
        "model": "alpha", "n_params": k_alpha, "n_data": n_data,
        "chi2_min": round(bf.fun, 2), "alpha": alpha, "Om": Om, "H0": H0,
        "M": M, "rd": rd,
        "chi2_lcdm_inscript": round(chi2_lcdm, 2),
        "chi2_w0wa": round(chi2_w0wa, 2),
        "dchi2_vs_lcdm": round(bf.fun - chi2_lcdm, 2),
    }])
    out.to_csv(os.path.join(_HERE, "dvch_alpha_fit.csv"), index=False)
    print("\nWrote dvch_alpha_fit.csv")


if __name__ == "__main__":
    main()
