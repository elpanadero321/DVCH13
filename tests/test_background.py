"""Numerical validation gates for the DVCH background solver (Appendix A tests)."""
import numpy as np
import pytest

from dvch import BackgroundSolver, DVCHParams
from dvch.growth import GrowthSolver

FID = DVCHParams(H0=70.0, Omega_m0=0.30, Omega_r0=9.0e-5, n=0.20, beta=1e-4)


@pytest.fixture(scope="module")
def bg():
    return BackgroundSolver(FID)


@pytest.fixture(scope="module")
def bg_lcdm():
    return BackgroundSolver(DVCHParams(H0=70.0, Omega_m0=0.30, Omega_r0=9.0e-5,
                                       n=0.0, beta=0.0))


def test_flatness_today(bg):
    """Omega_tot(0) = 1 within numerical tolerance; E(0) = 1."""
    assert abs(bg.E[0] - 1.0) < 1e-8
    assert abs(bg.Omega_r[0] + bg.Omega_m[0] + bg.Omega_L[0] - 1.0) < 1e-8


def test_closure_residual(bg):
    assert np.max(bg.closure_residual() / bg.E**2) < 1e-12


def test_conservation_residual(bg):
    """Eq. (21) residual small relative to 3 Om/(1+z)."""
    res = np.abs(bg.conservation_residual())
    scale = 3 * bg.Omega_m / (1 + bg.z)
    assert np.max(res[2:-2] / scale[2:-2]) < 1e-3  # numerical-gradient limited


def test_lcdm_limit_exact(bg_lcdm):
    """n = beta = 0 must reproduce analytic LCDM E(z) to solver precision."""
    p = bg_lcdm.p
    E2 = p.Omega_r0 * (1 + bg_lcdm.z) ** 4 + p.Omega_m0 * (1 + bg_lcdm.z) ** 3 + p.Omega_L0
    assert np.max(np.abs(bg_lcdm.E - np.sqrt(E2)) / np.sqrt(E2)) < 1e-8
    assert np.max(np.abs(bg_lcdm.Qtilde())) < 1e-12


def test_beta_zero_tracking_limit():
    """beta -> 0 reproduces the unsuppressed tracking closure OL = OL0 (Om/Om0)^n."""
    p = DVCHParams(n=0.2, beta=0.0)
    bg = BackgroundSolver(p)
    OL_expected = p.Omega_L0 * (bg.Omega_m / p.Omega_m0) ** p.n
    assert np.max(np.abs(bg.Omega_L - OL_expected)) < 1e-10


def test_Q_negative_fiducial(bg):
    """Fiducial run: Q < 0 for 0 <= z <= 5 (vacuum -> matter transfer)."""
    mask = bg.z <= 5.0
    assert np.all(bg.Qtilde()[mask] < 0.0)


def test_weff_sign_consistency(bg):
    """Eq. (30): Q < 0 <=> w_eff > -1 pointwise."""
    Qt, w = bg.Qtilde(), bg.w_eff()
    assert np.all((Qt < 0) == (w > -1.0))


def test_transition_redshift(bg):
    zt = bg.transition_redshift()
    assert zt is not None and 0.0 < zt < 2.0


def test_nec_and_gsl(bg):
    assert bg.nec_satisfied()


def test_bbn_gate(bg):
    assert bg.bbn_gate(0.04)


def test_ede_bound(bg):
    """f_EDE(1100) well below percent level (Eq. 28)."""
    assert bg.fEDE(1100.0) < 1e-4


def test_scan_n_viability():
    """Scan 0.05 <= n <= 0.45: H(z) > 0 everywhere (Appendix A gate iv).

    The additional 0 < zt < 2 viability gate of the manuscript scan is only
    satisfied for the lower part of the n range (viable fraction ~0.52 in the
    manuscript's (n,beta) grid); n <= 0.30 points must pass it here.
    """
    for n in np.linspace(0.05, 0.45, 9):
        b = BackgroundSolver(DVCHParams(n=float(n), beta=1e-4), z_max=100.0, n_grid=800)
        assert np.all(b.H > 0)
        if n <= 0.301:
            assert b.is_viable()


def test_growth_lcdm_sanity(bg_lcdm):
    """LCDM growth: f(0) ~ Omega_m^0.55 within ~1%."""
    g = GrowthSolver(bg_lcdm)
    f0 = float(g.f(0.0))
    assert abs(f0 - 0.30 ** 0.55) / 0.30 ** 0.55 < 0.02


def test_growth_dvch_finite(bg):
    g = GrowthSolver(bg)
    zz = np.linspace(0, 5, 50)
    assert np.all(np.isfinite(g.D(zz))) and np.all(np.isfinite(g.f(zz)))
    assert np.all(g.D(zz) > 0)


def test_distances_monotone(bg):
    dl = bg.luminosity_distance()
    assert np.all(np.diff(dl) > 0)
