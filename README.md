# DVCH — Dynamic Vacuum Coupling Hypothesis: Reproducible Pipeline

Reproducible numerical pipeline for the DVCH manuscript (`paper/DVCH.tex`).
All figures and tables are regenerated from code; no static manuscript image
is used as numerical evidence.

## Model conventions (fixed)

- `Q ≡ ρ̇_Λ` (Q < 0 = vacuum → matter transfer)
- `ρ̇_m + 3Hρ_m = −Q`, `ρ̇_Λ = Q`, `ρ̇_r + 4Hρ_r = 0`
- Closure: `Ω_Λ = Ω_Λ0 (Ω_m/Ω_m0)^n (1+β)/(1+βE²)`, `0 < n < 1`, `β = (H0/M_DVCH)²`
- Momentum transfer: `Q^μ = Q u_m^μ`

## Layout

```
dvch/            background solver (implicit Friedmann via Picard iteration,
                 exact kernel Q of Eq. 25, w_eff, distances) + sub-horizon growth
sympy_checks/    symbolic verification of all background equations (pytest)
tests/           numerical validation gates (flatness, ΛCDM limit, BBN/EDE/NEC
                 gates, (n,β) scan, growth sanity)
scripts/         figure/table regeneration (CSV + PNG + SHA256 checksums)
results/         machine-readable outputs (CSV, JSON summaries with checksums)
figures/         regenerated figures
class_dvch/      patched CLASS backend (phase 3)
cobaya/          Cobaya likelihood configurations (phase 5)
chains/          MCMC chains and checkpoints (phase 6)
```

## Installation (from scratch)

Tested on Ubuntu 22.04, Python 3.10.

```bash
python3 -m pip install -r requirements.txt
```

## Running

```bash
# 1. Symbolic verification of every background equation and the exact Q kernel
python3 -m pytest sympy_checks/ -v

# 2. Numerical validation gates
python3 -m pytest tests/ -v

# 3. Regenerate background tables and figures (deterministic, seed recorded)
python3 scripts/make_background_outputs.py
```

Outputs and their SHA256 checksums are written to
`results/background_summary.json`.

## Reproducibility

- Random seed: recorded in `results/background_summary.json` (`20260730`).
- All derived CSV/PNG artifacts carry SHA256 checksums.
- Exact package versions: `requirements.txt`; CLASS version/commit will be
  pinned in `class_dvch/VERSION` (phase 3).

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Symbolic verification (sympy) of Eqs. 6–8, 16, 19, 21–22, 25/B3, 30, 33, 39–49, 18, 27–28, 56–60 | done (29 tests pass) |
| 2 | Background solver + validation gates + regenerated figures | done |
| 3 | DVCH in pinned CLASS (background + full relativistic perturbations, δQ incl. δH) | pending |
| 4 | ΛCDM-limit (n=β=0) and synchronous vs Newtonian gauge validation | pending |
| 5 | Cobaya + Planck 2018 TT/TE/EE+lowE+lensing, Pantheon+, DESI BAO, RSD | pending |
| 6 | ≥4 chains to R̂−1<0.01, constraints on H0, Ωm0, n, β, σ8, S8 | pending |
| 7 | χ²/AIC/BIC + Bayesian evidence (PolyChord) + robustness | pending |
| 8 | Validation report + updated manuscript | pending |
