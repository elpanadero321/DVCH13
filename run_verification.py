import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from uvhde_cosmology import UVHDEPlanckCosmology


def E2_lcdm(z, Om=0.3153, Or=9.18e-5):
    ODE0 = 1.0 - Om - Or
    zp1 = 1.0 + np.asarray(z, dtype=np.float64)
    return Or * zp1**4 + Om * zp1**3 + ODE0


def main():
    uv = UVHDEPlanckCosmology()
    c_kms = uv.c_kms
    H0 = uv.H0
    Om = uv.Omega_m0
    Or = uv.Omega_r0
    rd = uv.rd

    print("=================================================================")
    print("ETAPA 1: FONDO COSMOLÓGICO Y REBOTE CUÁNTICO DE PLANCK")
    print("=================================================================")
    z_cross = brentq(lambda z: float(uv.w_eff(z)) + 1.0, 0.0, 2.0)
    z_acc = brentq(lambda z: float(uv.q(z)), 0.0, 2.0)
    t0 = (
        quad(lambda z: 1.0 / ((1.0 + z) * np.sqrt(uv.E2_classical(z))), 0.0, np.inf)[0]
        * (977.79222168 / H0)
    )
    print(f"w_eff(0)          = {float(uv.w_eff(0.0)):.4f}")
    print(f"Cruce w=-1        = z = {z_cross:.4f}")
    print(f"Inicio acelerac.  = z = {z_acc:.4f}")
    print(f"Edad del Universo = {t0:.3f} Gyr")
    print(f"eps_Pl            = {uv.eps_Pl:.4e}")
    print(f"Rebote cuántico   = a_min = {uv.a_min:.4e} (z_bounce = {uv.z_bounce:.4e})")

    print("\n=================================================================")
    print("ETAPA 2: DIAGNÓSTICO Om(z) (EN LCDM ES RECTA PLANA = 0.3154)")
    print("=================================================================")
    for z in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        om_u = uv.Om_diagnostic(z)
        om_l = Om if z == 0 else (E2_lcdm(z, Om, Or) - 1.0) / ((1.0 + z) ** 3 - 1.0)
        print(f"z = {z:4.1f} | Om_UVHDE = {om_u:.4f} vs Om_LCDM = {om_l:.4f} ({(om_u - om_l)/om_l*100:+6.2f}%)")

    print("\n=================================================================")
    print("ETAPA 3: DISTANCIAS BAO Y SUPERNOVAS (CALIBRADO AL CMB DE PLANCK)")
    print("=================================================================")
    omega_m = Om * (H0 / 100.0) ** 2
    omega_r = Or * (H0 / 100.0) ** 2
    z_star = 1089.92
    chi_target = quad(lambda z: 1.0 / np.sqrt(E2_lcdm(z, Om, Or)), 0.0, z_star)[0] / H0

    H0_cmb = brentq(
        lambda h0: quad(
            lambda z: 1.0
            / np.sqrt(
                UVHDEPlanckCosmology(
                    h0, omega_m / (h0 / 100.0) ** 2, omega_r / (h0 / 100.0) ** 2
                ).E2_classical(z)
            ),
            0.0,
            z_star,
        )[0]
        / h0
        - chi_target,
        60.0,
        75.0,
    )
    Om_cmb = omega_m / (H0_cmb / 100.0) ** 2
    Or_cmb = omega_r / (H0_cmb / 100.0) ** 2
    uv_cmb = UVHDEPlanckCosmology(H0=H0_cmb, Omega_m0=Om_cmb, Omega_r0=Or_cmb, rd=rd)
    print(f"Calibrado a Planck theta_*: H0 = {H0_cmb:.2f} km/s/Mpc, Omega_m0 = {Om_cmb:.4f}")

    for name, z in [("BGS", 0.295), ("LRG1", 0.510), ("LRG2", 0.706), ("LRG3+ELG1", 0.930), ("ELG2", 1.317), ("Lya", 2.330)]:
        dv_u = uv_cmb.D_V(z) / rd
        dm_l = quad(lambda zp: c_kms / (H0 * np.sqrt(E2_lcdm(zp, Om, Or))), 0.0, z)[0] / rd
        dh_l = (c_kms / (H0 * np.sqrt(E2_lcdm(z, Om, Or)))) / rd
        dv_l = (z * (dm_l**2) * dh_l) ** (1.0 / 3.0)
        dmu = 5.0 * np.log10((uv_cmb.D_M(z) / rd) / dm_l)
        print(f"{name:10s} (z={z:5.3f}) | DV/rd: UV={dv_u:6.2f} vs LCDM={dv_l:6.2f} ({(dv_u-dv_l)/dv_l*100:+5.2f}%) | Delta_mu={dmu:+.4f} mag")

    print("\n=================================================================")
    print("ETAPA 4: CRECIMIENTO LINEAL DE ESTRUCTURAS f*sigma8(z)")
    print("=================================================================")
    z_growth = np.array([0.38, 0.51, 0.61, 1.48])
    _, fs8_u = uv.growth_factor_and_fsigma8(z_growth, sigma8_0=0.8190)

    N_ini = np.log(1.0 / 1001.0)
    def ode_l(N, y):
        z = np.exp(-N) - 1.0
        zp1 = 1.0 + z
        e2 = E2_lcdm(z, Om, Or)
        de2 = 4.0 * Or * zp1**3 + 3.0 * Om * zp1**2
        dlnE = -0.5 * (zp1 / e2) * de2
        Om_a = Om * (zp1**3) / e2
        return [y[1], -(2.0 + dlnE) * y[1] + 1.5 * Om_a * y[0]]

    sol_l = solve_ivp(ode_l, [N_ini, 0.0], [np.exp(N_ini), np.exp(N_ini)], dense_output=True, rtol=1e-11, atol=1e-13)
    D0_l = sol_l.sol(0.0)[0]
    for z_val, f_u in zip(z_growth, fs8_u):
        Dl, dDl = sol_l.sol(-np.log(1.0 + z_val))
        f_l = (dDl / Dl) * 0.8110 * (Dl / D0_l)
        print(f"z = {z_val:4.2f} | fs8_UVHDE = {f_u:.4f} vs fs8_LCDM = {f_l:.4f} ({(f_u - f_l)/f_l*100:+5.2f}%)")


if __name__ == "__main__":
    main()
