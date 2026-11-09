import os
import sys

sys.path.insert(0, "/mnt/d/DVCH13-1")
os.chdir("/mnt/d/DVCH13-1")
import numpy as np

from alpha_cobaya_planck import AlphaPlanckLikelihood
import dvch_vs_lcdm_real as dvr

plk = AlphaPlanckLikelihood()
plk.initialize()
camb = plk._camb
params = plk._params(67.36, 0.02237, 0.1200, 0.054, 3.04, 0.965, 0.0)
res = camb.get_results(params)
der = res.get_derived_params()
rd = float(der["rdrag"])
print("rd =", rd)

# CC
c2 = 0.0
for z, H_obs, sigma in dvr.cc_data[:3]:
    hz = float(res.h_of_z(np.array([float(z)]))[0]) * 299792.458
    print(f"  CC z={z}: CAMB H={hz:.2f} obs={H_obs} sigma={sigma}")
    c2 += ((hz - H_obs) / sigma) ** 2
print("chi2_cc(3 pts) =", c2)

# BAO
for tracer, z, obs_list, corr in dvr.desi_data[:2]:
    z = float(z)
    dm = float(res.comoving_radial_distance(np.array([z]))[0])
    dh = 299792.458 / (float(res.h_of_z(np.array([z]))[0]) * 299792.458) * 299792.458
    print(f"  BAO {tracer} z={z}: D_M={dm:.1f} D_H={dh:.1f} rd={rd:.2f} "
          f"DM/rd={dm/rd:.3f} obs={[o[1] for o in obs_list]}")

# SNe
z_hd, z_hel, mb, dmb, cov, cinv = dvr.load_pantheon()
dl = np.array(res.luminosity_distance(np.asarray(z_hd[:5], dtype=float)))
print("dl first 5:", np.round(dl, 1))
mu = 5.0 * np.log10(dl) + 25.0 - 19.4
print("mu first 5:", np.round(mu, 3), "mb:", mb[:5])
