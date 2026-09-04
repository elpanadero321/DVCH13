"""Run the short real Cobaya chain against CAMB-DVCH and Planck plik."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from cobaya.run import run
from dvch_cobaya_planck import planck_loglike


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("DVCH_CAMB_ROOT", "/mnt/d/DVCH-external/CAMB-master")
os.environ.setdefault(
    "DVCH_CLIK_EGG",
    "/home/danieproyect/clik-install/local/lib/python3.14/dist-packages/"
    "clik-3.1-py3.14-linux-x86_64.egg",
)
os.environ.setdefault(
    "DVCH_PLANCK_LIKELIHOOD",
    "/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/hi_l/plik/"
    "plik_rd12_HM_v22b_TTTEEE.clik",
)


if __name__ == "__main__":
    info = yaml.safe_load((ROOT / "dvch_cobaya_short.yaml").read_text())
    info["likelihood"]["dvch_plik"]["external"] = planck_loglike
    info, sampler = run(info, debug=True, force=True)
    print("Cobaya short chain completed:", info["output"])
