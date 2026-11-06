#!/usr/bin/env python3
"""
Prueba de CONTROL: w0waCDM (energia oscura evolutiva) sobre los mismos
datos reales, con el mismo metodo que DVCH/LCDM.

Motivo cientifico: DESI DR1/DR2 + Pantheon+ ya prefieren w0waCDM frente a
LCDM por una cantidad similar. Si DVCH solo iguala a w0waCDM, esta
reproduciendo un fenomeno conocido. Si lo supera, es un resultado nuevo.

Usa los mismos datos (CC 31 puntos, DESI DR1 con correlaciones, Pantheon+
con corte z>0.01 y sin calibradoras) y el mismo prior plano marginalizado
sobre r_d que dvch_vs_lcdm_real.py. Incluye tests de robustez por subset
(SNe solo, BAO solo, CC solo).
"""
import numpy as np
import dvch_vs_lcdm_real as m
from dvch_vs_lcdm_real import load_pantheon, desi_data, cc_data

sne = load_pantheon()
Z_SN, Z_HEL, MB, DMB, COV, CINV = sne
print(f"Pantheon+: {len(Z_SN)} SNe (z>0.01, sin calibradoras)\n")


# ------------------------------------------------------------------
# w0waCDM: E2 = Or(1+z)^4 + Om(1+z)^3 + OL*(1+z)^(3(1+w0+wa))*exp(-3wa z/(1+z))
# ------------------------------------------------------------------
def E2_w0wa(z, Om, w0, wa):
    OL = 1.0 - Om - m.Omega_r0
    if OL <= 0:
        return np.nan
    opz = 1.0 + np.asarray(z, dtype=float)
    exponent = 3.0 * (1.0 + w0 + wa)
    de = OL * opz**exponent * np.exp(-3.0 * wa * (opz - 1.0) / opz)
    return m.Omega_r0 * opz**4 + Om * opz**3 + de


def _bao_w0wa(Om, H0, rd, w0, wa):
    e2g = E2_w0wa(m._ZGRID, Om, w0, wa)
    if np.any(np.isnan(e2g)) or np.any(e2g <= 0):
        return 1e10
    inv_e = 1.0 / np.sqrt(e2g)
    cum = np.concatenate([[0.0], np.cumsum((inv_e[1:] + inv_e[:-1]) * 0.5
                                           * np.diff(m._ZGRID))])
    chi2 = 0.0
    for tracer, z, obs_list, corr in desi_data:
        z = float(z)
        e2 = float(E2_w0wa(z, Om, w0, wa))
        if np.isnan(e2) or e2 <= 0:
            return 1e10
        chi_z = float(np.interp(z, m._ZGRID, cum))
        d_m = m.c_speed / H0 * chi_z
        d_h = m.c_speed / (H0 * np.sqrt(e2))
        if len(obs_list) == 2:
            th = np.array([d_m / rd, d_h / rd])
            obs = np.array([obs_list[0][1], obs_list[1][1]])
            s_m, s_h = obs_list[0][2], obs_list[1][2]
            det = s_m**2 * s_h**2 * (1.0 - corr**2)
            if det <= 0:
                return 1e10
            cinv = np.array([[s_h**2, -corr * s_m * s_h],
                             [-corr * s_m * s_h, s_m**2]]) / det
            d = th - obs
            chi2 += float(d @ cinv @ d)
        else:
            d_v = (z * d_m**2 * d_h) ** (1.0 / 3.0)
            _, val, sigma = obs_list[0]
            chi2 += ((d_v / rd - val) / sigma) ** 2
    return chi2


def chi2_all_w0wa(Om, H0, M, w0, wa, rd, subsets=("cc", "bao", "sne")):
    total = 0.0
    if "cc" in subsets:
        for z, H_obs, sigma in cc_data:
            e2 = E2_w0wa(z, Om, w0, wa)
            if np.isnan(e2) or e2 <= 0:
                return 1e10
            total += ((H0 * np.sqrt(e2) - H_obs) / sigma) ** 2
    if "bao" in subsets:
        total += _bao_w0wa(Om, H0, rd, w0, wa)
    if "sne" in subsets:
        e2g = E2_w0wa(m._ZGRID, Om, w0, wa)
        if np.any(np.isnan(e2g)) or np.any(e2g <= 0):
            return 1e10
        inv_e = 1.0 / np.sqrt(e2g)
        cum = np.concatenate([[0.0], np.cumsum((inv_e[1:] + inv_e[:-1]) * 0.5
                                               * np.diff(m._ZGRID))])
        integ = np.interp(Z_SN, m._ZGRID, cum)
        dl = (1.0 + Z_HEL) * m.c_speed / H0 * integ
        if np.any(dl <= 0):
            return 1e10
        mu = 5.0 * np.log10(dl) + 25.0 + M
        r = MB - mu
        total += float(r @ CINV @ r)
    return total


BOUNDS = [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0),
          (-3.0, 1.0), (-3.0, 2.0), m.RD_BOUNDS]
NAMES = ["Om", "H0", "M", "w0", "wa", "rd"]


def run_nested(subsets, label, nlive=400):
    import dynesty

    def logpost(p):
        Om, H0, M, w0, wa, rd = p
        for (lo, hi), v in zip(BOUNDS, p):
            if not (lo < v < hi):
                return -np.inf
        c2 = chi2_all_w0wa(Om, H0, M, w0, wa, rd, subsets)
        if c2 > 1e9:
            return -np.inf
        return -0.5 * c2

    def ptform(u):
        return np.array([lo + (hi - lo) * ui
                         for ui, (lo, hi) in zip(u, BOUNDS)])

    print(f"  (corriendo nested sampling: {label})")
    sampler = dynesty.NestedSampler(logpost, ptform, len(BOUNDS), nlive=nlive,
                                    sample="rslice", bound="multi",
                                    bootstrap=0)
    sampler.run_nested(print_progress=False)
    res = sampler.results
    from dynesty import utils as dyfunc
    weights = np.exp(res.logwt - res.logwt[-1])
    means = dyfunc.mean_and_cov(res.samples, weights)[0]
    print(f"\n=== {label} ===")
    print(f"  log Z = {res.logz[-1]:.3f} +/- {res.logzerr[-1]:.3f}")
    for nm, val in zip(NAMES, means):
        print(f"  {nm:6s} = {val:.4f}")
    return res.logz[-1], res.logzerr[-1], means


def main():
    print("########## MODELO DE CONTROL: w0waCDM ##########\n")
    r = run_nested(("cc", "bao", "sne"), "w0waCDM (todos los datos)")
    with open("realdata/logz_w0wa_cache.txt", "w") as fh:
        fh.write(f"{r[0]} {r[1]}")


if __name__ == "__main__":
    main()
