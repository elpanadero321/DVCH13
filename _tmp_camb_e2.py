import os
import sys

sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
import numpy as np
import camb

base = dict(H0=67.5, ombh2=0.0224, omch2=0.122, tau=0.054, As=2e-9, ns=0.965)
p = camb.set_params(**base, DVCH_flag=True, DVCH_n=0.2, DVCH_beta=1e-4)
p.set_cosmology(H0=67.5, ombh2=0.0224, omch2=0.122, num_massive_neutrinos=0,
                mnu=0.0, neutrino_hierarchy="degenerate")
r = camb.get_results(p)
zs = np.array([0.0, 1.0, 2.33, 3.0, 5.0, 7.0, 10.0, 20.0, 100.0])
hz = np.array(r.h_of_z(zs))
print("z      E2_camb")
for z, h in zip(zs, hz):
    print(f"{z:6.2f} {(h / hz[0])**2:10.4g}")
