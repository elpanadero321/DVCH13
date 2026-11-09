"""Paso 0 (segun INSTRUCCIONES_PARA_IA.md): ajusta el modelo alpha en el CMB
usando CAMB ESTANDAR con DE desacoplada en modo PPF (sin tocar el CAMB
parcheado). El fondo del modelo alpha es exactamente

    rho_DE(a) = OL + alpha*(1-a)^2

Modos (env ALPHA_MODE):
  - "lcdm"  : alpha fijada a 0 (CONTROL: mismo camino ppf, debe reproducir
              el chi2_min de LCDM ~ 2766.8 con nuisance oficial libre).
  - "free"  : alpha libre (prior [-2, 4], ref 0.6).

Usa el mismo yaml de nuisance oficial que run_dvch_cobaya_lcdm.py (las
priors oficiales se importan de ahi) y la misma pila de likelihoods de
Planck (high-l TTTEEE + lowl TT + lowE + lensing si env.sh los define).
"""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np
import yaml
from cobaya.run import run

from alpha_cobaya_planck import alpha_planck_loglike


ROOT = Path(__file__).resolve().parent

_REQUIRED_ENV = {
    "DVCH_CAMB_ROOT": "root del CAMB (parcheado; DVCH_flag por defecto False)",
    "DVCH_CLIK_EGG": "path al clik egg",
    "DVCH_PLANCK_LIKELIHOOD": "path a plik_rd12_HM_v22b_TTTEEE.clik",
    "DVCH_PLANCK_LOWL": "path a commander_xx.clik",
    "DVCH_PLANCK_LOWE": "path a simall_xx.clik",
    "DVCH_PLANCK_LENSING": "path a smicadx12_xx.clik_lensing",
}

_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    raise SystemExit(
        "Missing required environment variables:\n  "
        + "\n  ".join(f"{k}  ({_REQUIRED_ENV[k]})" for k in _missing)
        + "\n\nCopia env.sh.example a env.sh, edita las rutas, `source env.sh`."
    )

LIKELIHOOD = Path(os.environ["DVCH_PLANCK_LIKELIHOOD"])
EGG = Path(os.environ["DVCH_CLIK_EGG"])

if not os.environ.get("LD_LIBRARY_PATH"):
    raise SystemExit(
        "LD_LIBRARY_PATH no esta definido (debe apuntar al directorio con "
        "libclik.so). Corre via el launcher script."
    )

sys.path.insert(0, str(EGG))

# Priors oficiales de plik: misma fuente unica que el runner LCDM (E2-style:
# un solo archivo de la verdad, sin duplicar a mano).
from run_dvch_cobaya_lcdm import (  # noqa: E402
    OFFICIAL_PRIORS_NORM,
    OFFICIAL_PRIORS_UNIFORM,
    OFFICIAL_EXTRA_PRIOR_SZ,
    OFFICIAL_FIXED,
)

BASE_NAMES = ("H0", "ombh2", "omch2", "tau", "ln10As", "ns", "alpha")
MODE = os.environ.get("ALPHA_MODE", "lcdm")
FIX_ALPHA = float(os.environ.get("ALPHA_FIX", "0.0"))


def _alpha_loglike(**kwargs):
    if MODE == "lcdm":
        kwargs.pop("alpha", None)
        return alpha_planck_loglike(
            kwargs.pop("H0"), kwargs.pop("ombh2"), kwargs.pop("omch2"),
            kwargs.pop("tau"), kwargs.pop("ln10As"), kwargs.pop("ns"),
            alpha=FIX_ALPHA, **kwargs,
        )
    return alpha_planck_loglike(
        kwargs.pop("H0"), kwargs.pop("ombh2"), kwargs.pop("omch2"),
        kwargs.pop("tau"), kwargs.pop("ln10As"), kwargs.pop("ns"),
        kwargs.pop("alpha"), **kwargs,
    )


def build_info() -> dict:
    import clik  # type: ignore
    import clik.hpy as hpy  # type: ignore

    likelihood = clik.clik(str(LIKELIHOOD))
    check = np.asarray(hpy.File(str(LIKELIHOOD), "r")["clik/check_param"][:])
    names = tuple(likelihood.extra_parameter_names)
    expected = 7527 + len(names)
    if check.shape != (expected,):
        raise ValueError(
            f"Unexpected clik vector shape {check.shape}, expected {(expected,)}")

    info = yaml.safe_load(
        (ROOT / "alpha_cobaya_short.yaml").read_text()
    )
    info["likelihood"]["alpha_plik"]["external"] = _alpha_loglike

    if MODE == "lcdm":
        # LCDM anidado: alpha se elimina del muestreo y se inyecta fijo.
        info["params"].pop("alpha", None)

    accepted = BASE_NAMES if MODE != "lcdm" else BASE_NAMES[:-1]
    _alpha_loglike.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in accepted + names
        ]
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

    ref_file = os.environ.get("ALPHA_REF_MIN")
    if ref_file and os.path.exists(ref_file):
        refs = {}
        for line in open(ref_file):
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    refs[parts[2]] = float(parts[1])
                except ValueError:
                    continue
        for name, value in refs.items():
            if name in info["params"] and isinstance(
                    info["params"].get(name), dict) and "prior" in info["params"][name]:
                info["params"][name]["ref"] = {
                    "dist": "norm", "loc": value,
                    "scale": max(abs(value) * 0.02, 1e-4),
                }
        print(f"[ref] arrancando desde {ref_file}")

    info["sampler"] = {"minimize": {
        "ignore_prior": os.environ.get(
            "ALPHA_IGNORE_PRIOR", "true").lower() == "true",
    }}

    out = os.environ.get("ALPHA_CHAIN_OUTPUT",
                         "alpha_lcdm_min" if MODE == "lcdm" else "alpha_free_min")
    info["output"] = out
    return info


def main() -> None:
    debug = "--debug" in sys.argv
    resume = os.environ.get("ALPHA_RESUME", "false").lower() == "true"
    run(build_info(), debug=debug, resume=resume, force=not resume)


if __name__ == "__main__":
    main()
