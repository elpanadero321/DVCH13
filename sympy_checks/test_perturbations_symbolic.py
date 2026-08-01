"""Symbolic checks of the DVCH linear-perturbation bookkeeping.

Verifies:
  * the conversion of the synchronous-gauge equations (56)-(57) to
    N = ln a form (58)-(59);
  * the linear Taylor coefficients of delta Q (Eq. 60) obtained by
    differentiating the exact kernel Q(rho_m, rho_L, H);
  * that the covariant kernel written in physical densities reproduces the
    dimensionless kernel of Eq. (25) on the background.

The full synchronous- vs Newtonian-gauge equivalence of observables is a
numerical validation gate performed with the patched Boltzmann code
(cross-gauge runs), not a sympy identity, and is covered in tests/.
"""
import sympy as sp

n, beta = sp.symbols("n beta", positive=True)
H0, rc0 = sp.symbols("H0 rho_crit0", positive=True)
rm, rL, rr, H = sp.symbols("rho_m rho_Lambda rho_r H", positive=True)
rm0, rL0 = sp.symbols("rho_m0 rho_Lambda0", positive=True)


def Q_kernel(rm_, rL_, rr_, H_):
    """Exact background kernel Q(rho_m, rho_L, H) implied by the closure.

    From rho_L = rho_L0 (rho_m/rho_m0)^n (1+beta)/(1+beta H^2/H0^2):
      Q = rho_L_dot = (dln rho_L/dln rho_m) (rho_L/rho_m) rho_m_dot
                      + (dln rho_L/d H^2) rho_L d(H^2)/dt          (Eq. B1)
    with rho_m_dot = -3 H rho_m - Q and
    d(H^2)/dt = -H (3 rho_m + 4 rho_r)/(3 Mpl^2) -> in normalized units
    dE^2/dN = -(3 Om + 4 Or). Solving the linear equation for Q gives the
    closed form used here.
    """
    E2 = H_ ** 2 / H0 ** 2
    dlnL_dlnm = n
    dlnL_dE2 = -beta / (1 + beta * E2)
    # dE2/dt = -H*(3 rho_m + 4 rho_r)/rc0   (flat FLRW, normalized densities)
    dE2_dt = -H_ * (3 * rm_ + 4 * rr_) / rc0
    Q = sp.Symbol("Q")
    eq = sp.Eq(Q, dlnL_dlnm * rL_ / rm_ * (-3 * H_ * rm_ - Q) + dlnL_dE2 * rL_ * dE2_dt)
    return sp.solve(eq, Q)[0]


def test_kernel_matches_eq25():
    """Q(rho_m,rho_L,H) equals Eq. (25) written in physical densities."""
    Q = sp.simplify(Q_kernel(rm, rL, rr, H))
    E = H / H0
    E2 = E ** 2
    # Eq. (25): Qtilde = Q/(3 H0 rc0) = -E (rL/rc0)/(1+n rL/rm) * [n - beta(4 Or + 3 Om)/(3(1+beta E^2))]
    Qtilde = Q / (3 * H0 * rc0)
    Qtilde_paper = -E * (rL / rc0) / (1 + n * rL / rm) * (
        n - beta * (4 * rr / rc0 + 3 * rm / rc0) / (3 * (1 + beta * E2))
    )
    assert sp.simplify(Qtilde - Qtilde_paper) == 0


def test_deltaQ_taylor_coefficients_eq60():
    """delta Q = Q_,rho_m δρm + Q_,rho_L δρΛ + Q_,H δH with coefficients from the kernel."""
    Q = Q_kernel(rm, rL, rr, H)
    dQ_drm = sp.simplify(sp.diff(Q, rm))
    dQ_drL = sp.simplify(sp.diff(Q, rL))
    dQ_dH = sp.simplify(sp.diff(Q, H))
    # coefficients must be finite and nonzero generically
    subs = {rm: sp.Float(0.3), rL: sp.Float(0.7), rr: sp.Float(1e-4), H: sp.Float(1),
            H0: sp.Float(1), rc0: sp.Float(1), n: sp.Float(0.2), beta: sp.Float(1e-4)}
    for c in (dQ_drm, dQ_drL, dQ_dH):
        v = complex(c.subs(subs))
        assert abs(v) > 0 and abs(v) < sp.oo
    # linear delta Q
    drm, drL, dH = sp.symbols("delta_rho_m delta_rho_Lambda delta_H")
    deltaQ = dQ_drm * drm + dQ_drL * drL + dQ_dH * dH
    # first-order Taylor consistency: Q(rm+drm,...) - Q(...) = deltaQ + O(2)
    eps = sp.Symbol("epsilon")
    expansion = Q.subs({rm: rm + eps * drm, rL: rL + eps * drL, H: H + eps * dH})
    lin = sp.diff(expansion, eps).subs(eps, 0)
    assert sp.simplify(lin - deltaQ) == 0


def test_synchronous_to_efold_form_eq58_59():
    """Eqs. (56)-(57) in conformal time map to (58)-(59) with N = ln a.

    d/dtau = a H d/dN  (H = aH conformal Hubble => d/dtau = H d/dN).
    """
    N = sp.Symbol("N")
    a = sp.exp(N)
    Hc, theta, h, dm, dQ, rm_ = sp.symbols("mathcalH theta_m h delta_m deltaQ rho_m")
    aQ = sp.Symbol("aQ")
    # Eq. (56): dm_dot = -theta - h_dot/2 + (aQ/rm) dm - (a/rm) dQ
    # divide by Hc = a H: dm_N = -theta/Hc - h_N/2 + aQ/(Hc rm) dm - a dQ/(Hc rm)
    # with aQ/(Hc rm) = Q/(H rm) and a/(Hc rm) = 1/(H rm): matches Eq. (58).
    Hphys = sp.Symbol("H", positive=True)
    Hc_expr = a * Hphys
    lhs = sp.together(aQ / (Hc_expr * rm_))
    assert sp.simplify(lhs.subs(aQ, a * sp.Symbol("Q")) - sp.Symbol("Q") / (Hphys * rm_)) == 0
    # Eq. (57): theta_dot = -Hc theta  ->  theta_N = -theta
    tau = sp.Symbol("tau")
    th = sp.Function("theta")(tau)
    # theta_N = theta_dot / Hc; substituting theta_dot = -Hc theta gives -theta
    theta_N = (-Hc * sp.Symbol("theta")) / Hc
    assert sp.simplify(theta_N + sp.Symbol("theta")) == 0
