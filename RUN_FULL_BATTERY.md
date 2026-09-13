# RUN_FULL_BATTERY.md — Reproducing the complete DVCH battery

> Turnkey guide to run the **full DVCH validation battery** and the **full Planck high-l MCMC** end to end on a capable Linux machine.
> This document **complements** `README_INSTALL.md` (the authoritative install reference) and `env.sh.example`. It adds the exact
> offline-package integration steps so the pipeline can be built from the dependency archives shipped by the author, and it records
> the operational fixes already present in the repo (covmat handling, production launcher).
>
> **Integrity note (repo policy).** Never report priors, spectra, posteriors, or convergence numbers that were not produced by an
> actual converged run. This guide only tells you *how* to run; it does not assert any scientific result.

---

## 0. Why a dedicated machine is required

The full physics run needs components that cannot be installed in a restricted or offline sandbox:

- Internet access (or pre-staged wheels) for `scipy`, `cython`, `mpi4py`, `emcee`, `getdist`, `cobaya`, `camb`, `classy`.
- A Fortran toolchain (`gfortran`) and MPI (`mpirun`) to build the patched CAMB and the Planck `clik`/`plik` likelihood.
- Adequate hardware: recommended **>= 8 physical cores** and **>= 16 GB RAM** for the Planck high-l TTTEEE run.

If any of these is missing, only the pure-Python late-time battery (CC + BAO) will run; the Planck high-l MCMC will not.

---

## 1. Dependency archives

Create a `pkgs/` directory next to the repository and place the author-provided archives there:

| Archive | Purpose |
| --- | --- |
| `cobaya-3.6.2.zip` | Cobaya sampler (offline install) |
| `class_public-master.zip` | CLASS Boltzmann code (`classy`) |
| `COM_Likelihood_Code-v3.0_R3.01.tar.gz` | Planck 2018 likelihood code (`clik`/`plik`) |
| `COM_Likelihood_Data-extra-lensing-ext_R3.00.tar.gz` | Planck low-l + lensing likelihood data |
| `DataRelease-main.zip` | Planck high-l `plik` data (**required** for the full high-l run) |

---

## 2. Build and install (offline friendly)

> `README_INSTALL.md` is authoritative for exact versions and paths. The steps below only show how to consume the archives above.

```bash
# 2.1 Isolated environment
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel

# 2.2 Core scientific stack (needs internet or a local wheelhouse)
pip install -r requirements.txt
pip install -r requirements-cmb.txt   # scipy, cython, mpi4py, emcee, getdist, camb, classy

# 2.3 Cobaya from the shipped archive
unzip -q pkgs/cobaya-3.6.2.zip -d build/
pip install ./build/cobaya-3.6.2

# 2.4 CLASS (classy) from the shipped archive
unzip -q pkgs/class_public-master.zip -d build/
make -C build/class_public-master -j"$(nproc)"
(cd build/class_public-master/python && pip install .)

# 2.5 Planck likelihood CODE (clik/plik) via its own waf build
tar xf pkgs/COM_Likelihood_Code-v3.0_R3.01.tar.gz -C build/
cd build/code/plc_3.0/plc-3.01
./waf configure --install_all_deps
./waf install
cd -

# 2.6 Planck likelihood DATA (low-l + lensing)
tar xf pkgs/COM_Likelihood_Data-extra-lensing-ext_R3.00.tar.gz -C data/

# 2.7 Planck high-l plik DATA
unzip -q pkgs/DataRelease-main.zip -d data/
```

Build the patched CAMB fork as documented in `README_INSTALL.md` (requires `gfortran`). The DVCH background model source is
`camb_dvch_model.f90` / `dvch_camb_background.py`; the cross-check lives in `dvch_camb_crosscheck.py`.

---

## 3. Configure the environment

```bash
cp env.sh.example env.sh
# Edit env.sh and set at least:
#   DVCH_CLIK_LIB_DIR, DVCH_CLIK_EGG, DVCH_CAMB_ROOT
#   DVCH_PLANCK_LIKELIHOOD, DVCH_PLANCK_LOWL, DVCH_PLANCK_LOWE, DVCH_PLANCK_LENSING
#   LD_LIBRARY_PATH  (must include the clik lib dir)
source env.sh
```

---

## 4. Preflight (fail fast before the long run)

```bash
python dvch_planck_preflight.py        # checks paths, data, and importability
python dvch_planck_clik_smoke.py       # clik self-check; compare against the values in README_INSTALL.md
```

Do not start the full MCMC until preflight and the clik smoke test pass.

---

## 5. Run the full validation battery

Unit + physics battery (late-time CC + DESI DR1 BAO, background, growth/sigma8, CAMB cross-check, joint real-data fit,
double-slit congruence). On Windows the orchestrator is `run_full_battery.ps1`; on Linux run the equivalent steps:

```bash
python -m pytest tests/ -q                 # unit tests
python dvch_camb_crosscheck.py             # background cross-check vs CAMB
python dvch_growth_diagnostic.py           # growth / sigma8, S8
python dvch_joint_realdata_fit.py          # joint CC + BAO fit (chi2/AIC/BIC)
python dvch_double_slit.py                 # double-slit congruence check
python dvch_verify_doc_numbers.py          # verify the numbers quoted in the paper match the CSV outputs
```

---

## 6. Full Planck high-l MCMC (production)

Use the production launcher, which already **separates the input covmat from the chain output prefix** so Cobaya cleanup cannot
delete the covmat it is reading:

```bash
./launch_prod.sh
#   DVCH_COVMAT=dvch_prod_wide.covmat     (input; distinct prefix)
#   DVCH_CHAIN_OUTPUT=dvch_prod           (output prefix)
# Tunables: DVCH_CHAIN_SEED, DVCH_MAX_SAMPLES, DVCH_BURN_IN,
#           DVCH_LEARN_PROPOSAL, DVCH_RMINUS1_STOP
```

Entry point: `run_dvch_cobaya_full_highl.py` (config `dvch_cobaya_short.yaml` + likelihood wrapper `dvch_cobaya_planck.py`).
Monitor convergence with:

```bash
python dvch_planck_chain_diagnostics.py    # target: R-hat < 1.01 AND ESS > 100 for all sampled params
```

The run is considered converged only when the diagnostic criterion is met. Report results **only** from a converged run.

---

## 7. Reproducibility checklist

- [ ] `requirements*.txt` installed; `cobaya`, `classy`, `camb` import cleanly.
- [ ] `clik` builds and `dvch_planck_clik_smoke.py` matches the README self-check values.
- [ ] Patched CAMB compiled; `dvch_camb_crosscheck.py` passes.
- [ ] Preflight passes.
- [ ] Full high-l MCMC reaches R-hat < 1.01 and ESS > 100.
- [ ] `dvch_verify_doc_numbers.py` confirms the manuscript numbers match the produced CSVs.
