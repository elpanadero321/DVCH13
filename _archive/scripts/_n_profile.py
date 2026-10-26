"""Perfil de chi2 de Planck vs DVCH_n (beta fijo pequeño)."""
import os
import sys

os.environ.setdefault("DVCH_CAMB_ROOT", "/mnt/d/DVCH-external/CAMB-master")
os.environ.setdefault("DVCH_CLIK_EGG", "/home/danieproyect/plc-3.1/clik-egg")
os.environ.setdefault("DVCH_CLIK_LIB_DIR", "/home/danieproyect/plc-3.1/lib")
os.environ.setdefault(
    "DVCH_PLANCK_LIKELIHOOD",
    "/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/hi_l/plik/"
    "plik_rd12_HM_v22b_TTTEEE.clik",
)
os.environ.setdefault(
    "DVCH_PLANCK_LOWL",
    "/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/low_l/commander/"
    "commander_dx12_v3_2_29.clik",
)
os.environ.setdefault(
    "DVCH_PLANCK_LOWE",
    "/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/low_l/simall/"
    "simall_100x143_offlike5_EE_Aplanck_B.clik",
)
os.environ.setdefault(
    "DVCH_PLANCK_LENSING",
    "/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/lensing/"
    "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing",
)
os.environ["LD_LIBRARY_PATH"] = os.environ["DVCH_CLIK_LIB_DIR"]
sys.path.insert(0, "/mnt/d/DVCH13-1")

import numpy as np
from dvch_cobaya_planck import planck_loglike

BASE = dict(H0=67.277, ombh2=0.022378, omch2=0.12017, tau=0.0561,
            ln10As=3.0485, ns=0.9659)

print("Perfil loglike vs DVCH_n (beta=1e-4 fijo, cosmologia fija)")
print(f"\n{'n':>8} {'loglike':>12} {'chi2':>12} {'dchi2':>10}")
ref = None
for n in [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25]:
    try:
        ll = planck_loglike(DVCH_n=n, DVCH_beta=1e-4, **BASE)
        if ref is None:
            ref = ll
        chi2 = -2.0 * ll
        print(f"{n:>8.4g} {ll:>12.4f} {chi2:>12.4f} {-2*(ll-ref):>10.4f}")
    except Exception as e:
        print(f"{n:>8.4g} -> {type(e).__name__}: {e}")
