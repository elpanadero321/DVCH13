"""Symbolic verification of the DVCH background equations (manuscript conventions).

Conventions verified (equation numbers refer to the DVCH manuscript):
  Q = rho_Lambda_dot                                (Eq. 8)
  rho_m_dot + 3 H rho_m = -Q                        (Eq. 7)
  rho_r_dot + 4 H rho_r = 0                         (Eq. 6)
  Omega_L = Omega_L0 (Om/Om0)^n (1+beta)/(1+beta E^2)   (Eq. 16)
  E^2 = Or + Om + OL                                (Eq. 12/19)
  Qtilde = Q/(3 H0 rho_crit0) = -E(1+z)/3 dOL/dz    (Eq. 24)
  Exact kernel (Eq. 25 / B3):
    Qtilde = -E OL / (1 + n OL/Om) * [ n - beta (4 Or0 (1+z)^4 + 3 Om) / (3 (1+beta E^2)) ]
"""
import sympy as sp

z, n, beta, Or0, Om0, OL0 = sp.symbols("z n beta Omega_r0 Omega_m0 Omega_Lambda0", positive=True)


def _setup():
    """Return (Om, OL, E2, dOm, dOL) with derivatives solved from conservation.

    Om(z) is implicit; its derivative dOm/dz is fixed by the total dark-sector
    conservation law dOm/dz + dOL/dz = 3 Om/(1+z) (Eq. 21) together with the
    chain rule applied to the closure (Eq. 16).
    """
    Om = sp.Function("Omega_m")(z)
    E2 = sp.Function("E2")(z)
    OL = OL0 * (Om / Om0) ** n * (1 + beta) / (1 + beta * E2)

    dOm = sp.Symbol("dOm")  # unknown dOm/dz
    # dE^2/dz is exact (Eq. 22): dE^2/dz = 4 Or0 (1+z)^3 + 3 Om/(1+z)
    dE2 = 4 * Or0 * (1 + z) ** 3 + 3 * Om / (1 + z)
    # chain rule on the closure
    dOL = sp.diff(OL, Om) * dOm + sp.diff(OL, E2) * dE2
    # conservation (Eq. 21)
    sol = sp.solve(sp.Eq(dOm + dOL, 3 * Om / (1 + z)), dOm)
    assert len(sol) == 1
    dOm_sol = sp.simplify(sol[0])
    dOL_sol = sp.simplify(dOL.subs(dOm, dOm_sol))
    return Om, OL, E2, dOm_sol, dOL_sol


def test_total_conservation_identity():
    """(6)+(7)+(8) => rho_tot_dot + 3H(rho_m + 4 rho_r/3) = 0 identically."""
    H, rm, rr, rL, Q = sp.symbols("H rho_m rho_r rho_Lambda Q")
    rm_dot = -3 * H * rm - Q
    rL_dot = Q
    rr_dot = -4 * H * rr
    total = rm_dot + rr_dot + rL_dot + 3 * H * (rm + sp.Rational(4, 3) * rr)
    assert sp.simplify(total) == 0


def test_exact_Q_kernel_eq25():
    """Derived Qtilde equals the closed-form kernel of Eq. (25)/(B3)."""
    Om, OL, E2, dOm, dOL = _setup()
    E = sp.sqrt(E2)
    Qtilde_derived = -E * (1 + z) / 3 * dOL
    bracket = n - beta * (4 * Or0 * (1 + z) ** 4 + 3 * Om) / (3 * (1 + beta * E2))
    Qtilde_paper = -E * OL / (1 + n * OL / Om) * bracket
    assert sp.simplify(Qtilde_derived - Qtilde_paper) == 0


def test_dE_dz_eq22():
    """Eq. (22): dE/dz = (4 Or0 (1+z)^3 + 3 Om/(1+z)) / (2E), from Eq. 21."""
    Om, OL, E2, dOm, dOL = _setup()
    dE2 = dOm + dOL + 4 * Or0 * (1 + z) ** 3  # d(Or+Om+OL)/dz
    expected = 4 * Or0 * (1 + z) ** 3 + 3 * Om / (1 + z)
    assert sp.simplify(dE2 - expected) == 0


def test_weff_identity_eq30():
    """Eq. (30): w_eff = -1 - Q/(3 H rho_L), from Eq. (29) with rho_L_dot = Q."""
    H, rL, Q = sp.symbols("H rho_Lambda Q", positive=True)
    weff = sp.symbols("w_eff")
    sol = sp.solve(sp.Eq(Q + 3 * H * (1 + weff) * rL, 0), weff)[0]
    assert sp.simplify(sol - (-1 - Q / (3 * H * rL))) == 0


def test_scalar_reconstruction_eq33():
    """Eq. (33): phidot^2 = (1+w_eff) rho_L = -Q/(3H); Q<0 => ghost-free."""
    H, rL, Q = sp.symbols("H rho_Lambda Q", positive=True)
    weff = -1 - Q / (3 * H * rL)
    phidot2 = (1 + weff) * rL
    assert sp.simplify(phidot2 - (-Q / (3 * H))) == 0


def test_autonomous_system_beta0_eq39_42():
    """beta->0 closure y = C x^n gives Q/(H rho_crit0) = -3 n x y/(x+n y) (Eq. 39)
    and the reduced system (40)-(42)."""
    x = sp.Function("x")(sp.Symbol("N"))
    N = sp.Symbol("N")
    C = sp.Symbol("C", positive=True)
    x_ = sp.Function("x")(N)
    y_ = C * x_ ** n
    # conservation in N: x' + y' = -3x  (from Eqs. 36-38 summed, radiation separate)
    xp = sp.Symbol("xp")
    yp = sp.diff(y_, x_) * xp
    sol = sp.solve(sp.Eq(xp + yp, -3 * x_), xp)[0]
    # Q/(H rho_crit0) = y' = -x' - 3x
    Qhat = sp.simplify(sp.diff(y_, x_) * sol)
    expected = -3 * n * x_ * y_ / (x_ + n * y_)
    assert sp.simplify(Qhat - expected) == 0
    # x' = -3x - Qhat = -3x + 3 n x y/(x+n y) = -3x^2/(x+ny)
    assert sp.simplify(sol - (-3 * x_ ** 2 / (x_ + n * y_))) == 0


def test_jacobian_eq45_46():
    """Jacobian of the reduced field (44) and its matter-direction limit (46)."""
    x, y, u = sp.symbols("x y u", positive=True)
    f1 = -3 * x ** 2 / (x + n * y)
    f2 = -3 * n * x * y / (x + n * y)
    f3 = -4 * u
    J = sp.Matrix([[sp.diff(f, v) for v in (x, y, u)] for f in (f1, f2, f3)])
    J_paper = sp.Matrix([
        [-3 * x * (x + 2 * n * y) / (x + n * y) ** 2, 3 * n * x ** 2 / (x + n * y) ** 2, 0],
        [-3 * n ** 2 * y ** 2 / (x + n * y) ** 2, -3 * n * x ** 2 / (x + n * y) ** 2, 0],
        [0, 0, -4],
    ])
    assert sp.simplify(J - J_paper) == sp.zeros(3, 3)
    Jm = sp.simplify(J.subs(y, 0))
    assert Jm == sp.Matrix([[-3, 3 * n, 0], [0, -3 * n, 0], [0, 0, -4]])
    eigs = set(Jm.eigenvals().keys())
    assert eigs == {-3, -3 * n, -4}


def test_lyapunov_eq48_49():
    """V = x+y+u gives V' = -3x - 4u <= 0 for x,u >= 0 (Eqs. 48-49)."""
    x, y, u = sp.symbols("x y u", nonnegative=True)
    f1 = -3 * x ** 2 / (x + n * y)
    f2 = -3 * n * x * y / (x + n * y)
    f3 = -4 * u
    Vp = sp.simplify(f1 + f2 + f3)
    assert sp.simplify(Vp - (-3 * x - 4 * u)) == 0


def test_small_beta_expansion_eq18():
    """Eq. (18): OL ~ OL0 (Om/Om0)^n [1 + beta(1-E^2) + O(beta^2 E^4)]."""
    Om, E2s = sp.symbols("Omega_m E2", positive=True)
    OL = OL0 * (Om / Om0) ** n * (1 + beta) / (1 + beta * E2s)
    series = sp.series(OL, beta, 0, 2).removeO()
    expected = OL0 * (Om / Om0) ** n * (1 + beta * (1 - E2s))
    assert sp.simplify(series - expected) == 0


def test_ede_scaling_eq27_28():
    """Eq. (27)-(28): OL/Om ~ (OL0/Om0)(1+z)^{3(n-1)}; numeric check at z=1100."""
    ratio = (OL0 / Om0) * (1 + z) ** (3 * (n - 1))
    val = ratio.subs({OL0: sp.Float(2.17), Om0: 1, n: sp.Float(0.2), z: 1099})
    fEDE = float(val)
    assert abs(fEDE - 2.17 * 1100 ** (-2.4)) / fEDE < 1e-12
    assert fEDE < 2e-7  # "well below percent-level EDE limits"


def test_bracket_sign_small_beta():
    """For beta*E^2 << 1 the bracket in Eq. (25) -> n > 0, hence Q < 0."""
    Om = sp.Symbol("Omega_m", positive=True)
    E2s = sp.Symbol("E2", positive=True)
    bracket = n - beta * (4 * Or0 * (1 + z) ** 4 + 3 * Om) / (3 * (1 + beta * E2s))
    assert sp.simplify(bracket.subs(beta, 0) - n) == 0
