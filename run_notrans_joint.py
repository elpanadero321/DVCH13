"""Cierre final: la formula del paper SIN transferencia (energia oscura
evolutiva pura, materia intacta) en el test CONJUNTO Planck + DESI DR1 +
Pantheon+ + CC.

El vacio del paper se traduce a un w(a) exacto:
    rho_L(a) = OL x^n (1+beta) / (1 + beta E2),  x = (1+z)^3
    E2 resuelto por la cuadratica exacta (materia = Om a^-3, sin Q).
    w(a) = -1 - (a/3) dln rho_L/dlna
y se pasa a CAMB estandar con set_dark_energy_w_a (ppf). Ninguna
transferencia: la materia es a^-3 y el vacio se conserva como DE.

Control de comparacion: alpha_joint_lcdm_min.bestfit.txt (chi2 = 4178.59),
la misma pila de likelihoods.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from cobaya.run import run

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from alpha_cobaya_planck import AlphaPlanckLikelihood  # noqa: E402
import dvch_vs_lcdm_real as dvr  # noqa: E402

_required = ["DVCH_CAMB_ROOT", "DVCH_CLIK_EGG", "DVCH_PLANCK_LIKELIHOOD",
             "DVCH_PLANCK_LOWL", "DVCH_PLANCK_LOWE", "DVCH_PLANCK_LENSING"]
_missing = [k for k in _required if not os.environ.get(k)]
if _missing:
    raise SystemExit(f"Faltan variables de entorno: {_missing}. source env.sh")

EGG = Path(os.environ["DVCH_CLIK_EGG"])
sys.path.insert(0, str(EGG))

from run_dvch_cobaya_lcdm import (  # noqa: E402
    OFFICIAL_PRIORS_NORM, OFFICIAL_PRIORS_UNIFORM, OFFICIAL_EXTRA_PRIOR_SZ,
    OFFICIAL_FIXED)

C_KMS = 299792.458
OR0 = dvr.Omega_r0

_PLK: AlphaPlanckLikelihood | None = None
_SNE = None


def _plk():
    global _PLK
    if _PLK is None:
        _PLK = AlphaPlanckLikelihood()
        _PLK.initialize()
    return _PLK


def _sne():
    global _SNE
    if _SNE is None:
        _SNE = dvr.load_pantheon()
    return _SNE


def w_de_table(H0, ombh2, omch2, n, beta):
    """Tabla (a, w) del vacio del paper SIN transferir."""
    h = H0 / 100.0
    Om = (ombh2 + omch2) / h ** 2
    OL = 1.0 - Om - OR0
    a = np.geomspace(1e-5, 1.0, 400)
    opz = 1.0 / a
    B = OR0 * opz ** 4 + Om * opz ** 3
    A = OL * opz ** (3 * n) * (1.0 + beta)
    b = 1.0 - beta * B
    disc = np.maximum(b * b + 4.0 * beta * (A + B), 0.0)
    e2 = np.maximum(0.0, -b + np.sqrt(disc)) / (2.0 * beta)
    rho = A / (1.0 + beta * e2)
    if np.any(rho <= 0):
        return None
    dl = np.gradient(np.log(rho), np.log(a))
    w = -1.0 - a / 3.0 * rho * dl / rho * 0  # placeholder, reemplazado abajo
    # derivada central de ln rho en ln a
    lr = np.log(rho)
    la = np.log(a)
    d = np.gradient(lr, la)
    w = -1.0 - d / 3.0
    return a, w


def _camb_params(H0, ombh2, omch2, tau, ln10As, ns, n, beta):
    camb = _plk()._camb
    As = float(np.exp(ln10As) / 1.0e10)
    params = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, As=As, ns=ns,
        lmax=2508, lens_potential_accuracy=1,
        WantTensors=False, DoLensing=True)
    if beta > 1e-9:
        tab = w_de_table(H0, ombh2, omch2, n, beta)
        if tab is None:
            return None
        a, w = tab
        if not np.all(np.isfinite(w)):
            return None
        params.set_dark_energy_w_a(a, w, dark_energy_model="ppf")
    return params


def chi2_late(H0, ombh2, omch2, n, beta, Msn):
    camb = _plk()._camb
    params = _camb_params(H0, ombh2, omch2, 0.054, 3.04, 0.965, n, beta)
    if params is None:
        return 1e10
    res = camb.get_results(params)
    der = res.get_derived_params()
    rd = float(der["rdrag"])
    chi2 = 0.0
    for z, H_obs, sigma in dvr.cc_data:
        hz = float(res.h_of_z(np.array([float(z)]))[0]) * C_KMS
        chi2 += ((hz - H_obs) / sigma) ** 2
    for tracer, z, obs_list, corr in dvr.desi_data:
        z = float(z)
        d_m = float(res.comoving_radial_distance(np.array([z]))[0])
        h_kms = float(res.h_of_z(np.array([z]))[0]) * C_KMS
        d_h = C_KMS / h_kms
        if len(obs_list) == 2:
            th = np.array([d_m / rd, d_h / rd])
            obs = np.array([obs_list[0][1], obs_list[1][1]])
            s_m, s_h = obs_list[0][2], obs_list[1][2]
            det = s_m ** 2 * s_h ** 2 * (1.0 - corr ** 2)
            cinv = np.array([[s_h ** 2, -corr * s_m * s_h],
                             [-corr * s_m * s_h, s_m ** 2]]) / det
            d = th - obs
            chi2 += float(d @ cinv @ d)
        else:
            d_v = (z * d_m ** 2 * d_h) ** (1.0 / 3.0)
            _, val, sigma = obs_list[0]
            chi2 += ((d_v / rd - val) / sigma) ** 2
    z_hd, z_hel, mb, dmb, cov, cinv = _sne()
    dl = np.array(res.luminosity_distance(np.asarray(z_hd, dtype=float)))
    if np.any(dl <= 0):
        return 1e10
    mu = 5.0 * np.log10(dl) + 25.0 + Msn
    r = mb - mu
    chi2 += float(r @ cinv @ r)
    return chi2


def _joint_loglike(H0, ombh2, omch2, tau, ln10As, ns, n, beta, Msn,
                   **nuisance):
    if not (0.01 < n < 1.0 and 0.0 < beta < 10.0):
        return -np.inf
    lp = _plk().logp(H0, ombh2, omch2, tau, ln10As, ns, 0.0, **nuisance)
    if not np.isfinite(lp):
        return -np.inf
    if os.environ.get("NOTRANS_CMB_ONLY", "false").lower() == "true":
        return lp
    c2 = chi2_late(H0, ombh2, omch2, n, beta, Msn)
    if c2 > 1e9:
        return -np.inf
    return lp - 0.5 * c2


def _lcdm_joint_loglike(H0, ombh2, omch2, tau, ln10As, ns, Msn, **nuisance):
    return _joint_loglike(H0, ombh2, omch2, tau, ln10As, ns, 0.01, 0.0, Msn,
                          **nuisance)


def build_info() -> dict:
    import clik  # type: ignore
    import clik.hpy as hpy  # type: ignore

    likelihood = clik.clik(str(os.environ["DVCH_PLANCK_LIKELIHOOD"]))
    check = np.asarray(
        hpy.File(str(os.environ["DVCH_PLANCK_LIKELIHOOD"]),
                 "r")["clik/check_param"][:])
    names = tuple(likelihood.extra_parameter_names)
    expected = 7527 + len(names)
    if check.shape != (expected,):
        raise ValueError(f"clik vector shape {check.shape}")

    info = yaml.safe_load((_HERE / "notrans_joint.yaml").read_text())
    mode = os.environ.get("NOTRANS_MODE", "free")
    if mode == "lcdm":
        info["likelihood"]["notrans_joint"]["external"] = _lcdm_joint_loglike
        for k in ("n", "beta"):
            info["params"].pop(k, None)
        accepted = ["H0", "ombh2", "omch2", "tau", "ln10As", "ns", "Msn"]
    else:
        info["likelihood"]["notrans_joint"]["external"] = _joint_loglike
        accepted = ["H0", "ombh2", "omch2", "tau", "ln10As", "ns", "n",
                    "beta", "Msn"]
    fn = info["likelihood"]["notrans_joint"]["external"]
    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(x, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for x in list(accepted) + list(names)]
    )

    for index, name in enumerate(names):
        value = float(check[7527 + index])
        if name in OFFICIAL_PRIORS_NORM:
            loc, scale = OFFICIAL_PRIORS_NORM[name]
            info["params"][name] = {
                "prior": {"dist": "norm", "loc": loc, "scale": scale},
                "ref": {"dist": "norm", "loc": loc, "scale": scale}}
        elif name in OFFICIAL_PRIORS_UNIFORM:
            lo, hi, ref_loc, ref_scale = OFFICIAL_PRIORS_UNIFORM[name]
            info["params"][name] = {
                "prior": {"dist": "uniform", "min": lo, "max": hi},
                "ref": {"dist": "norm", "loc": ref_loc, "scale": ref_scale}}
        elif name in OFFICIAL_FIXED:
            info["params"][name] = {"value": OFFICIAL_FIXED[name]}
        else:
            info["params"][name] = {"value": value}
    if {"ksz_norm", "A_sz"}.issubset(set(names)):
        info.setdefault("prior", {})["SZ"] = OFFICIAL_EXTRA_PRIOR_SZ

    info["sampler"] = {"minimize": {
        "ignore_prior": os.environ.get(
            "NOTRANS_IGNORE_PRIOR", "true").lower() == "true"}}
    info["output"] = os.environ.get(
        "NOTRANS_CHAIN_OUTPUT",
        "notrans_lcdm_min" if mode == "lcdm" else "notrans_free_min")
    return info


def main() -> None:
    debug = "--debug" in sys.argv
    run(build_info(), debug=debug, force=True)


if __name__ == "__main__":
    main()
