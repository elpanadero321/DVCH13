"""Run LCDM (the n=0, beta=0 nested limit of DVCH) for model comparison.

LCDM is recovered exactly from the DVCH closure:
    OL_z = OL * (Om_z/Om)^n * (1+beta) / (1 + beta*E2)
With n=0, beta=0: OL_z = OL, i.e. pure LCDM. We keep DVCH_flag=True but
pin both DVCH parameters to zero, so the SAME patched CAMB and the SAME
likelihood stack produce the LCDM chi2 on identical data (Planck 2018
TTTEEE + low-l). That makes chi2_DVCH - chi2_LCDM a valid nested-model
comparison.

Two modes, selected by env var DVCH_MODE:
  - "minimize": cobaya Minimizer -> MAP / min chi2. Fast (~1-3 h).
  - "sample":   full MCMC for marginal posteriors of LCDM (~6-9 days).
"""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np
import yaml
from cobaya.run import run

from dvch_cobaya_planck import planck_loglike


ROOT = Path(__file__).resolve().parent

_REQUIRED_ENV = {
    "DVCH_CAMB_ROOT": "root directory of the patched CAMB checkout",
    "DVCH_CLIK_EGG": "path to the clik Python egg (clik-3.1-py3.x-*.egg)",
    "DVCH_PLANCK_LIKELIHOOD": "path to plik_rd12_HM_v22b_TTTEEE.clik",
    "DVCH_PLANCK_LOWL": "path to commander_dx12_v3_2_29.clik",
    "DVCH_PLANCK_LOWE": "path to simall_100x143_offlike5_EE_Aplanck_B.clik",
    "DVCH_PLANCK_LENSING": "path to smicadx12_..._consext8.clik_lensing",
}

_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    raise SystemExit(
        "Missing required environment variables:\n  "
        + "\n  ".join(f"{k}  ({_REQUIRED_ENV[k]})" for k in _missing)
        + "\n\nCopy env.sh.example to env.sh, edit the paths, then `source env.sh`."
    )

LIKELIHOOD = Path(os.environ["DVCH_PLANCK_LIKELIHOOD"])
EGG = Path(os.environ["DVCH_CLIK_EGG"])

if not os.environ.get("LD_LIBRARY_PATH"):
    raise SystemExit(
        "LD_LIBRARY_PATH is not set. It must point to the directory containing "
        "libclik.so (e.g. /path/to/plc-3.1/lib). Run via the launcher script."
    )

sys.path.insert(0, str(EGG))

OFFICIAL_PRIORS = {
    "gal545_A_100": (8.6, 2.0),
    "gal545_A_143": (10.6, 2.0),
    "gal545_A_143_217": (23.5, 8.5),
    "gal545_A_217": (91.9, 20.0),
    "galf_EE_A_100": (0.055, 0.014),
    "galf_EE_A_100_143": (0.040, 0.010),
    "galf_EE_A_100_217": (0.094, 0.023),
    "galf_EE_A_143": (0.086, 0.022),
    "galf_EE_A_143_217": (0.21, 0.051),
    "galf_EE_A_217": (0.70, 0.18),
    "galf_TE_A_100": (0.13, 0.042),
    "galf_TE_A_100_143": (0.13, 0.036),
    "galf_TE_A_100_217": (0.46, 0.09),
    "galf_TE_A_143": (0.207, 0.072),
    "galf_TE_A_143_217": (0.69, 0.09),
    "galf_TE_A_217": (1.938, 0.54),
    "calib_100T": (1.0002, 0.0007),
    "calib_217T": (0.99805, 0.00065),
}

OFFICIAL_FIXED = {
    "A_planck": 1.000442,
    "cib_index": -1.3,
    "galf_EE_index": -2.4,
    "galf_TE_index": -2.4,
    "calib_100P": 1.021,
    "calib_143P": 0.966,
    "calib_217P": 1.040,
    "A_cnoise_e2e_100_100_EE": 1.0,
    "A_cnoise_e2e_143_143_EE": 1.0,
    "A_cnoise_e2e_217_217_EE": 1.0,
    "A_sbpx_100_100_TT": 1.0,
    "A_sbpx_143_143_TT": 1.0,
    "A_sbpx_143_217_TT": 1.0,
    "A_sbpx_217_217_TT": 1.0,
    "A_sbpx_100_100_EE": 1.0,
    "A_sbpx_100_143_EE": 1.0,
    "A_sbpx_100_217_EE": 1.0,
    "A_sbpx_143_143_EE": 1.0,
    "A_sbpx_143_217_EE": 1.0,
    "A_sbpx_217_217_EE": 1.0,
}

# Base cosmological params sampled in LCDM (DVCH_n and DVCH_beta are pinned).
BASE_NAMES = ("H0", "ombh2", "omch2", "tau", "ln10As", "ns")

# Valores fijos de n y beta: por defecto 0 (LCDM puro), pero pueden
# sobreescribirse con DVCH_FIX_N / DVCH_FIX_BETA para un perfil de chi2.
FIX_N = float(os.environ.get("DVCH_FIX_N", "0.0"))
FIX_BETA = float(os.environ.get("DVCH_FIX_BETA", "0.0"))


def _lcdm_loglike(**kwargs):
    """planck_loglike with DVCH_n and DVCH_beta pinned (0,0 = pure LCDM)."""
    cosm = {k: kwargs.pop(k) for k in BASE_NAMES}
    return planck_loglike(
        cosm["H0"], cosm["ombh2"], cosm["omch2"], cosm["tau"],
        cosm["ln10As"], cosm["ns"],
        DVCH_n=FIX_N, DVCH_beta=FIX_BETA,
        **kwargs,
    )


def build_info() -> dict:
    import clik  # type: ignore
    import clik.hpy as hpy  # type: ignore

    likelihood = clik.clik(str(LIKELIHOOD))
    check = np.asarray(hpy.File(str(LIKELIHOOD), "r")["clik/check_param"][:])
    names = tuple(likelihood.extra_parameter_names)
    expected = 7527 + len(names)
    if check.shape != (expected,):
        raise ValueError(f"Unexpected clik vector shape {check.shape}, expected {expected}")

    info = yaml.safe_load(
        (ROOT / os.environ.get("DVCH_SHORT_YAML", "dvch_cobaya_short.yaml")).read_text()
    )
    info["likelihood"]["dvch_plik"]["external"] = _lcdm_loglike

    # LCDM: los parametros DVCH se eliminan del muestreo (caso anidado con
    # n=0, beta=0). _lcdm_loglike los inyecta fijos a cero.
    info["params"].pop("DVCH_n", None)
    info["params"].pop("DVCH_beta", None)

    # Cobaya ve una likelihood cuyos argumentos aceptados son exactamente los
    # parametros muestreados de LCDM: 6 cosmologicos + los nuisance de plik.
    _lcdm_loglike.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in BASE_NAMES + names
        ]
    )

    for index, name in enumerate(names):
        value = float(check[7527 + index])
        if name in OFFICIAL_PRIORS:
            loc, scale = OFFICIAL_PRIORS[name]
            info["params"][name] = {
                "prior": {"dist": "norm", "loc": loc, "scale": scale},
                "ref": {"dist": "norm", "loc": loc, "scale": scale},
            }
        elif name in OFFICIAL_FIXED:
            info["params"][name] = {"value": OFFICIAL_FIXED[name]}
        else:
            info["params"][name] = {"value": value}

    mode = os.environ.get("DVCH_MODE", "minimize")
    if mode == "minimize":
        info["sampler"] = {"minimize": {}}
    else:
        info["sampler"] = {"mcmc": {
            "max_samples": int(os.environ.get("DVCH_MAX_SAMPLES", "32")),
            "burn_in": float(os.environ.get("DVCH_BURN_IN", "0")),
            "learn_proposal": (
                os.environ.get("DVCH_LEARN_PROPOSAL", "false").lower() == "true"
            ),
            "Rminus1_stop": float(os.environ.get("DVCH_RMINUS1_STOP", "0.01")),
        }}
        if os.environ.get("DVCH_LEARN_RMINUS1_MAX"):
            info["sampler"]["mcmc"]["learn_proposal_Rminus1_max"] = float(
                os.environ["DVCH_LEARN_RMINUS1_MAX"]
            )
        if os.environ.get("DVCH_LEARN_EVERY"):
            info["sampler"]["mcmc"]["learn_every"] = os.environ["DVCH_LEARN_EVERY"]
        if os.environ.get("DVCH_NO_BLOCKING", "").lower() in ("1", "true", "yes"):
            info["sampler"]["mcmc"].pop("blocking", None)
        if os.environ.get("DVCH_COVMAT"):
            info["sampler"]["mcmc"]["covmat"] = os.environ["DVCH_COVMAT"]
        if os.environ.get("DVCH_CHAIN_SEED"):
            info["sampler"]["mcmc"]["seed"] = int(os.environ["DVCH_CHAIN_SEED"])

    info["output"] = os.environ.get("DVCH_CHAIN_OUTPUT", "dvch_lcdm_chain")
    return info


def main() -> None:
    debug = "--debug" in sys.argv
    resume = os.environ.get("DVCH_RESUME", "false").lower() == "true"
    run(build_info(), debug=debug, resume=resume, force=not resume)


if __name__ == "__main__":
    main()
