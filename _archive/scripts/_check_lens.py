import os
import sys

sys.path.insert(0, "/home/danieproyect/plc-3.1/clik-egg")
import clik  # noqa: E402

lk = clik.clik_lensing(os.environ["DVCH_PLANCK_LENSING"])
print("lensing lmax:", tuple(int(x) for x in lk.lmax))
print("lensing nuisance:", lk.extra_parameter_names)
nm = tuple(lk.extra_parameter_names)
import numpy as np  # noqa: E402

print("n lmax entries:", len(lk.lmax))
