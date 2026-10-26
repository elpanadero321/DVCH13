"""Perfil de chi2 de Planck vs DVCH_beta usando el modulo del proyecto.

Mantiene cosmologia fija en los valores del best-fit encontrado y solo
varia DVCH_beta. Responde: ¿el CMB prefiere beta pequeno o el late-time?
"""
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

print("Perfil loglike vs DVCH_beta (n=0.07 fijo, cosmologia fija)")
print("Los nuisance toman los valores oficiales de check_param (default).")
print(f"\n{'beta':>8} {'loglike':>12} {'chi2':>12} {'dchi2':>10}")
ref = None
for beta in [1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    try:
        ll = planck_loglike(DVCH_n=0.07, DVCH_beta=beta, **BASE)
        if ref is None:
            ref = ll
        chi2 = -2.0 * ll
        print(f"{beta:>8.4g} {ll:>12.4f} {chi2:>12.4f} {-2*(ll-ref):>10.4f}")
    except Exception as e:
        print(f"{beta:>8.4g} -> {type(e).__name__}: {e}")
