"""Run the short real Cobaya chain against CAMB-DVCH and Planck plik."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from cobaya.run import run
from dvch_cobaya_planck import planck_loglike


ROOT = Path(__file__).resolve().parent

# Portable path resolution: read from environment (see env.sh.example).
_REQUIRED_ENV = {
    "DVCH_CAMB_ROOT": "root directory of the patched CAMB checkout",
    "DVCH_CLIK_EGG": "path to the clik Python egg (clik-3.1-py3.x-*.egg)",
    "DVCH_PLANCK_LIKELIHOOD": "path to plik_rd12_HM_v22b_TTTEEE.clik",
}
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    raise SystemExit(
        "Missing required environment variables:\n  "
        + "\n  ".join(f"{k}  ({_REQUIRED_ENV[k]})" for k in _missing)
        + "\n\nCopy env.sh.example to env.sh, edit the paths, then `source env.sh`."
    )


if __name__ == "__main__":
    info = yaml.safe_load((ROOT / "dvch_cobaya_short.yaml").read_text())
    info["likelihood"]["dvch_plik"]["external"] = planck_loglike
    debug = os.environ.get("DVCH_DEBUG", "false").lower() == "true"
    info, sampler = run(info, debug=debug, force=True)
    print("Cobaya short chain completed:", info["output"])
