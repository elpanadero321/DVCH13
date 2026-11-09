import sys

import numpy as np

sys.path.insert(0, "/mnt/d/DVCH13-1")
sys.path.insert(0, "/mnt/d/DVCH-external/CAMB-master")
from camb import get_results  # noqa: E402

import _diag_alpha_cmb as d  # noqa: E402


def main():
    OL = 1.0 - 0.02237 / 0.6736**2 - d.OMCH2 / 0.6736**2
    for alpha in (0.0, 0.597):
        res = get_results(d.params_for(67.36, alpha))
        der = res.get_derived_params()
        print(f"--- alpha={alpha} (OL={OL:.4f}) ---")
        rd = der["rdrag"]
        print(f"  r_drag={rd:.4f}  theta*={100*der['thetastar']:.4f}  "
              f"z_drag={der['zdrag']:.2f}")
        zs = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        hz = res.h_of_z(zs) * 299792.458
        for z, hv in zip(zs, hz):
            print(f"  H({z:>4}) = {hv:8.3f}")
        # densidad de DE que ve CAMB en la cuadratura del fondo
        for a_i in (1.0, 0.5, 1.0 / 1.5):
            bd = res.get_background_densities(np.array([a_i]))
            key = "grho_lambda" if "grho_lambda" in bd else list(bd)[2]
            grhov = bd[key][0]
            exp_rho = OL + alpha * (1 - a_i) ** 2
            print(f"  a={a_i:.4f}: {key}={grhov:.5f}  "
                  f"esperado={exp_rho:.5f}")


if __name__ == "__main__":
    main()
