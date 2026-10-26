import os
import sys

sys.path.insert(0, "/home/danieproyect/plc-3.1/clik-egg")
import clik  # noqa: E402

from run_dvch_cobaya_full_highl import OFFICIAL_PRIORS  # noqa: E402

paths = {
    "plik": os.environ["DVCH_PLANCK_LIKELIHOOD"],
    "lowl": os.environ["DVCH_PLANCK_LOWL"],
    "lowE": os.environ["DVCH_PLANCK_LOWE"],
    "lens": os.environ["DVCH_PLANCK_LENSING"],
}
sampled = set(OFFICIAL_PRIORS)
for tag, p in paths.items():
    try:
        lk = clik.clik_lensing(p) if tag == "lens" else clik.clik(p)
        nm = set(lk.extra_parameter_names)
        print(tag, "n=", len(nm), "overlap=", sorted(nm & sampled))
    except Exception as e:
        print(tag, "ERR", e)
