"""Package the DVCH pipeline into a portable zip for external execution.

Excludes:
  * Planck clik data (LICENSED — user must obtain from Planck Legacy Archive).
  * Run artifacts (chain .txt/.checkpoint/.covmat/.progress/.dill_pickle/
    .updated.yaml/.log files and result CSVs).
  * Python bytecode caches (__pycache__).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "DVCH_pipeline_portable.zip"

# Source files that are part of the pipeline (code, configs, docs, patch, tests).
INCLUDE_FILES = [
    # Core pipeline
    "dvch_boltzmann_backend.py",
    "dvch_camb_background.py",
    "dvch_modified_boltzmann_solver.py",
    "dvch_perturbations.py",
    "dvch_cobaya_planck.py",
    "dvch_full_mcmc_pipeline.py",
    "dvch_mcmc_convergence.py",
    "dvch_planck_chain_diagnostics.py",
    "dvch_realdata_diagnostics.py",
    "dvch_relativistic_perturbation_validation.py",
    "dvch_robustness_scan.py",
    "dvch_verify_doc_numbers.py",
    "dvch_planck_preflight.py",
    "dvch_planck_clik_smoke.py",
    "dvch_colab_simple.py",
    "dvch_joint_realdata_fit.py",
    "dvch_cmb_class_camb_mcmc.py",
    "dvch_double_slit.py",
    "dvch_growth_diagnostic.py",
    # Runners
    "run_dvch_cobaya_full_highl.py",
    "run_dvch_cobaya_full_highl.sh",
    "run_dvch_cobaya_short.py",
    "launch_prod.sh",
    # Configs
    "dvch_cobaya_short.yaml",
    "dvch_planck2018_full.yaml",
    "env.sh.example",
    "requirements.txt",
    "requirements-cmb.txt",
    # Docs
    "README_INSTALL.md",
    # CAMB patch
    "camb_patch/DVCHModel.f90",
    "camb_patch/README_patch.md",
    # Tests
    "tests/__init__.py",
    "tests/test_dvch.py",
]

# Reference data tables (small, non-licensed, useful for reproducibility).
INCLUDE_CSV = [
    "dvch_background_table.csv",
    "dvch_boltzmann_background.csv",
    "dvch_boltzmann_modified_background.csv",
    "dvch_boltzmann_modified_growth.csv",
    "dvch_boltzmann_modified_matter_power.csv",
    "dvch_boltzmann_source_table.csv",
    "dvch_growth_sigma8_table.csv",
]


def main() -> None:
    missing = [f for f in INCLUDE_FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit("Missing files:\n  " + "\n  ".join(missing))

    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_FILES + INCLUDE_CSV:
            zf.write(ROOT / rel, arcname=rel)
        # Write a manifest of what is included.
        zf.writestr(
            "MANIFEST.txt",
            "\n".join(INCLUDE_FILES + INCLUDE_CSV) + "\n",
        )

    size_kb = OUT.stat().st_size / 1024
    print(f"Created {OUT.name} ({size_kb:.1f} KB, {len(INCLUDE_FILES) + len(INCLUDE_CSV)} files)")


if __name__ == "__main__":
    main()