"""Paso 0 (completo): ajuste CONJUNTO Planck + DESI DR1 BAO + Pantheon+ + CC
del modelo alpha, usando CAMB estandar (ppf) solo de fondo.

Logica: una sola likelihood externa que suma
    chi2 = chi2_Planck(clik, con nuisance oficial)
         + chi2_BAO(DESI DR1, D_M/r_d y D_H/r_d con r_d del CAMB)
         + chi2_CC (31 H(z) con H(z) del CAMB)
         + chi2_SNe (Pantheon+ 1580, M muestreado)
Todo con el MISMO CAMB (mismo fondo) -> sin doble evaluation.

Parametros muestreados: H0, ombh2, omch2, tau, ln10As, ns, alpha, M
+ nuisance oficial de plik (priors oficiales, modo minimize ignore_prior).
Modos (ALPHA_MODE): "lcdm" (alpha=0) o "free".
"""
from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np
import yaml
from cobaya.run import run

from alpha_cobaya_planck import AlphaPlanckLikelihood
import dvch_vs_lcdm_real as dvr  # noqa: E402


ROOT = Path(__file__).resolve().parent

_required = ["DVCH_CAMB_ROOT", "DVCH_CLIK_EGG", "DVCH_PLANCK_LIKELIHOOD"]
_missing = [k for k in _required if not os.environ.get(k)]
if _missing:
    raise SystemExit(f"Faltan variables de entorno: {_missing}. source env.sh")

EGG = Path(os.environ["DVCH_CLIK_EGG"])
sys.path.insert(0, str(EGG))

from run_dvch_cobaya_lcdm import (  # noqa: E402
    OFFICIAL_PRIORS_NORM,
    OFFICIAL_PRIORS_UNIFORM,
    OFFICIAL_EXTRA_PRIOR_SZ,
    OFFICIAL_FIXED,
)

BASE_NAMES = ("H0", "ombh2", "omch2", "tau", "ln10As", "ns", "alpha", "Msn")
MODE = os.environ.get("ALPHA_MODE", "free")
FIX_ALPHA = float(os.environ.get("ALPHA_FIX", "0.0"))
C_KMS = 299792.458

_PLK: AlphaPlanckLikelihood | None = None
_SNE = None


def _planck():
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


def chi2_late_from_camb(H0, ombh2, omch2, alpha, M):
    """chi2 de CC + DESI BAO + Pantheon+ usando el fondo del CAMB ppf."""
    plk = _planck()
    camb = plk._camb
    params = plk._params(H0, ombh2, omch2, 0.054, 3.04, 0.965, alpha)
    results = camb.get_results(params)
    der = results.get_derived_params()
    rd = float(der["rdrag"])
    chi2 = 0.0
    # --- cosmic chronometers ---
    for z, H_obs, sigma in dvr.cc_data:
        hz = float(results.h_of_z(np.array([float(z)]))[0]) * C_KMS
        chi2 += ((hz - H_obs) / sigma) ** 2
    # --- DESI DR1 BAO (D_M/r_d, D_H/r_d) ---
    for tracer, z, obs_list, corr in dvr.desi_data:
        z = float(z)
        d_m = float(results.comoving_radial_distance(np.array([z]))[0])
        h_kms = float(results.h_of_z(np.array([z]))[0]) * C_KMS
        d_h = C_KMS / h_kms
        if len(obs_list) == 2:
            th = np.array([d_m / rd, d_h / rd])
            obs = np.array([obs_list[0][1], obs_list[1][1]])
            s_m, s_h = obs_list[0][2], obs_list[1][2]
            det = s_m**2 * s_h**2 * (1.0 - corr**2)
            cinv = np.array([[s_h**2, -corr * s_m * s_h],
                             [-corr * s_m * s_h, s_m**2]]) / det
            d = th - obs
            chi2 += float(d @ cinv @ d)
        else:
            d_v = (z * d_m**2 * d_h) ** (1.0 / 3.0)
            _, val, sigma = obs_list[0]
            chi2 += ((d_v / rd - val) / sigma) ** 2
    # --- Pantheon+ ---
    z_hd, z_hel, mb, dmb, cov, cinv = _sne()
    dl = np.array(results.luminosity_distance(np.asarray(z_hd, dtype=float)))
    if np.any(dl <= 0):
        return 1e10
    mu = 5.0 * np.log10(dl) + 25.0 + M
    r = mb - mu
    chi2 += float(r @ cinv @ r)
    return chi2


def _joint_loglike(H0, ombh2, omch2, tau, ln10As, ns, alpha, Msn, **nuisance):
    lp = _planck().logp(H0, ombh2, omch2, tau, ln10As, ns, alpha, **nuisance)
    if not np.isfinite(lp):
        return -np.inf
    c2_late = chi2_late_from_camb(H0, ombh2, omch2, alpha, Msn)
    if c2_late > 1e9:
        return -np.inf
    return lp - 0.5 * c2_late


def _lcdm_joint_loglike(H0, ombh2, omch2, tau, ln10As, ns, Msn, **nuisance):
    return _joint_loglike(H0, ombh2, omch2, tau, ln10As, ns, FIX_ALPHA, Msn,
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
        raise ValueError(f"clik vector shape {check.shape}, expected {expected}")

    info = yaml.safe_load((ROOT / "alpha_joint.yaml").read_text())
    if MODE == "lcdm":
        info["likelihood"]["alpha_joint"]["external"] = _lcdm_joint_loglike
        info["params"].pop("alpha", None)
        accepted = BASE_NAMES[:6] + BASE_NAMES[7:]  # sin alpha, con Msn
    else:
        info["likelihood"]["alpha_joint"]["external"] = _joint_loglike
        accepted = BASE_NAMES
    fn = info["likelihood"]["alpha_joint"]["external"]
    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in accepted + names]
    )

    for index, name in enumerate(names):
        value = float(check[7527 + index])
        if name in OFFICIAL_PRIORS_NORM:
            loc, scale = OFFICIAL_PRIORS_NORM[name]
            info["params"][name] = {
                "prior": {"dist": "norm", "loc": loc, "scale": scale},
                "ref": {"dist": "norm", "loc": loc, "scale": scale},
            }
        elif name in OFFICIAL_PRIORS_UNIFORM:
            lo, hi, ref_loc, ref_scale = OFFICIAL_PRIORS_UNIFORM[name]
            info["params"][name] = {
                "prior": {"dist": "uniform", "min": lo, "max": hi},
                "ref": {"dist": "norm", "loc": ref_loc, "scale": ref_scale},
            }
        elif name in OFFICIAL_FIXED:
            info["params"][name] = {"value": OFFICIAL_FIXED[name]}
        else:
            info["params"][name] = {"value": value}
    if {"ksz_norm", "A_sz"}.issubset(set(names)):
        info.setdefault("prior", {})["SZ"] = OFFICIAL_EXTRA_PRIOR_SZ

    info["sampler"] = {"minimize": {
        "ignore_prior": os.environ.get(
            "ALPHA_IGNORE_PRIOR", "true").lower() == "true",
    }}
    out = os.environ.get(
        "ALPHA_CHAIN_OUTPUT",
        "alpha_joint_lcdm_min" if MODE == "lcdm" else "alpha_joint_free_min")
    info["output"] = out
    return info


def main() -> None:
    debug = "--debug" in sys.argv
    run(build_info(), debug=debug, force=True)


if __name__ == "__main__":
    main()
