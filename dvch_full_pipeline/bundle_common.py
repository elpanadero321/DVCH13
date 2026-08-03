#!/usr/bin/env python3
"""
Shared helpers for the dedicated DVCH full-production bundle.

This folder keeps the heavy Planck + Pantheon+ + BAO Cobaya/classy pipeline
separate from the lighter diagnostic scripts stored at repository root.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

BUNDLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dvch_theory import DVCHParameters, build_classy_extra_args, fiducial_decay_summary

STATUS_DIR = BUNDLE_ROOT / "status"
CHAINS_DIR = BUNDLE_ROOT / "chains"
DEFAULT_OUTPUT_ROOT = CHAINS_DIR / "dvch_full"
YAML_PATH = BUNDLE_ROOT / "dvch_full_cobaya.yaml"
MANIFEST_PATH = BUNDLE_ROOT / "dvch_full_runtime_manifest.json"
STATUS_CSV = STATUS_DIR / "dvch_full_pipeline_status.csv"
STATUS_PNG = STATUS_DIR / "dvch_full_pipeline_status.png"
README_PATH = BUNDLE_ROOT / "README.txt"
EXAMPLE_PS1_PATH = BUNDLE_ROOT / "run_full_pipeline_example.ps1"

FULL_LIKELIHOOD_NAMES = [
    "planck_2018_highl_plik.TTTEEE",
    "planck_2018_lowl.TT_clik",
    "planck_2018_lowl.EE_clik",
    "planck_2018_lensing.clik",
    "sn.pantheonplus",
    "bao.sixdf_2011_bao",
    "bao.sdss_dr7_mgs",
    "bao.sdss_dr12_consensus_bao",
    "bao.desi_2024_bao_all",
]

OPTIONAL_EXTENSION_NAMES = [
    "custom cosmic chronometers likelihood",
    "custom local H0 Gaussian prior",
    "custom compressed fsigma8 / RSD likelihood",
]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_layout(output_root: Path | None = None) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    if output_root is not None:
        output_root.parent.mkdir(parents=True, exist_ok=True)


def build_likelihood_block() -> dict[str, dict[str, Any]]:
    return {name: {} for name in FULL_LIKELIHOOD_NAMES}


def build_params_block() -> dict[str, dict[str, Any]]:
    return {
        "omega_b": {
            "prior": {"min": 0.0200, "max": 0.0245},
            "ref": 0.02237,
            "proposal": 0.00010,
            "latex": r"\omega_b",
        },
        "omega_cdm": {
            "prior": {"min": 0.090, "max": 0.140},
            "ref": 0.1199,
            "proposal": 0.0015,
            "latex": r"\omega_{\rm cdm}",
        },
        "H0": {
            "prior": {"min": 60.0, "max": 80.0},
            "ref": 69.0,
            "proposal": 0.80,
            "latex": r"H_0",
        },
        "A_s": {
            "prior": {"min": 1.5e-9, "max": 2.8e-9},
            "ref": 2.10e-9,
            "proposal": 0.05e-9,
            "latex": r"A_s",
        },
        "n_s": {
            "prior": {"min": 0.92, "max": 1.02},
            "ref": 0.965,
            "proposal": 0.004,
            "latex": r"n_s",
        },
        "tau_reio": {
            "prior": {"min": 0.01, "max": 0.12},
            "ref": 0.054,
            "proposal": 0.007,
            "latex": r"\tau_{\rm reio}",
        },
        "dvch_n": {
            "prior": {"min": 0.01, "max": 0.40},
            "ref": 0.18,
            "proposal": 0.015,
            "latex": r"n_{\rm DVCH}",
        },
        "log10_dvch_beta": {
            "prior": {"min": -6.0, "max": -2.0},
            "ref": -4.0,
            "proposal": 0.12,
            "drop": True,
            "latex": r"\log_{10}\beta_{\rm DVCH}",
        },
        "dvch_beta": {
            "value": "lambda log10_dvch_beta: 10**log10_dvch_beta",
            "latex": r"\beta_{\rm DVCH}",
        },
        "S8": {
            "derived": "lambda sigma8, Omega_m: sigma8*(Omega_m/0.3)**0.5",
            "latex": r"S_8",
        },
    }


def build_full_info(
    *,
    classy_path: str | None,
    output_root: Path,
    gr_limit: float,
    max_samples: int,
) -> dict[str, Any]:
    return {
        "theory": {
            "classy": {
                "path": classy_path,
                "stop_at_error": True,
                "ignore_obsolete": True,
                "extra_args": build_classy_extra_args(
                    z_pk="0 0.5 1 2",
                    l_max_scalars=3500,
                    pk_max_h_mpc=5.0,
                ),
            }
        },
        "likelihood": build_likelihood_block(),
        "params": build_params_block(),
        "sampler": {
            "mcmc": {
                "drag": True,
                "learn_proposal": True,
                "learn_proposal_Rminus1_max": 2.0,
                "learn_every": "40d",
                "max_tries": "80d",
                "oversample_power": 0.4,
                "proposal_scale": 1.9,
                "Rminus1_stop": gr_limit,
                "Rminus1_cl_stop": gr_limit,
                "max_samples": max_samples,
            }
        },
        "output": str(output_root),
    }


def build_runtime_manifest(
    *,
    packages_path: str | None,
    classy_path: str | None,
    output_root: Path,
    gr_limit: float,
    max_samples: int,
    info: dict[str, Any],
    decay_summary: dict[str, float | bool],
    execution_state: str,
    gelman_rubin_rminus1: float | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "bundle_status": execution_state,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(BUNDLE_ROOT),
        "repo_root": str(REPO_ROOT),
        "paths": {
            "packages_path": packages_path or "<set-cobaya-packages-path>",
            "classy_path": classy_path or "<set-patched-classy-path>",
            "output_root": str(output_root),
            "yaml_path": str(YAML_PATH),
            "status_csv": str(STATUS_CSV),
            "status_png": str(STATUS_PNG),
            "example_launcher": str(EXAMPLE_PS1_PATH),
        },
        "required_python_packages": ["cobaya", "getdist", "classy"],
        "required_external_runtime": [
            "Patched CLASS/classy backend that accepts dvch_n, dvch_beta, dvch_model, and dvch_use_exact_q.",
            "Cobaya packages path containing Planck 2018 clik data, Pantheon+, and BAO likelihood files.",
        ],
        "core_likelihoods": FULL_LIKELIHOOD_NAMES,
        "optional_follow_up_likelihoods": OPTIONAL_EXTENSION_NAMES,
        "convergence_targets": {
            "Rminus1_stop": gr_limit,
            "Rminus1_cl_stop": gr_limit,
            "max_samples": max_samples,
        },
        "dvch_bridge_extra_args": info["theory"]["classy"]["extra_args"],
        "fiducial_decay_check": decay_summary,
        "notes": [
            "This folder is prepared for a future heavy run but is not executed automatically.",
            "Planck/CMB validation remains pending until the patched classy backend and the Cobaya likelihood data are installed.",
            "Root-level lightweight and intermediate scripts remain untouched and separate from this bundle.",
        ],
    }
    if gelman_rubin_rminus1 is not None:
        manifest["gelman_rubin_rminus1"] = gelman_rubin_rminus1
    return manifest


def build_status_rows(
    *,
    packages_path: str | None,
    classy_path: str | None,
    info: dict[str, Any],
    decay_summary: dict[str, float | bool],
    yaml_written: bool,
    manifest_written: bool,
) -> list[dict[str, Any]]:
    classy_path_ready = bool(classy_path) and Path(classy_path).exists()
    packages_path_ready = bool(packages_path) and Path(packages_path).exists()
    classy_ready = module_available("classy") or classy_path_ready
    selected_likelihoods = ", ".join(info["likelihood"].keys())

    return [
        {
            "name": "Dedicated bundle folder",
            "available": True,
            "type": "layout",
            "detail": f"Prepared under {BUNDLE_ROOT.name}.",
        },
        {
            "name": "Cobaya YAML export",
            "available": yaml_written,
            "type": "config",
            "detail": f"Written to {YAML_PATH.name}.",
        },
        {
            "name": "Runtime manifest",
            "available": manifest_written,
            "type": "config",
            "detail": f"Written to {MANIFEST_PATH.name}.",
        },
        {
            "name": "PowerShell launch template",
            "available": EXAMPLE_PS1_PATH.exists(),
            "type": "launcher",
            "detail": f"Stored as {EXAMPLE_PS1_PATH.name}.",
        },
        {
            "name": "Cobaya Python package",
            "available": module_available("cobaya"),
            "type": "sampler",
            "detail": "Needed to execute the production MCMC run.",
        },
        {
            "name": "GetDist post-processing",
            "available": module_available("getdist"),
            "type": "post",
            "detail": "Used to verify Gelman-Rubin convergence after the run.",
        },
        {
            "name": "Patched classy backend",
            "available": classy_ready,
            "type": "boltzmann",
            "detail": (
                "Supply --classy-path to a patched CLASS build, or install a patched "
                "classy module."
            ),
        },
        {
            "name": "Cobaya packages data path",
            "available": packages_path_ready,
            "type": "data",
            "detail": "Planck 2018, Pantheon+, and BAO data packages.",
        },
        {
            "name": "Selected likelihood modules",
            "available": True,
            "type": "likelihood",
            "detail": selected_likelihoods,
        },
        {
            "name": "DVCH fiducial decay check",
            "available": bool(
                decay_summary["vacuum_decays_monotonically"]
                and decay_summary["positive_energy_densities"]
                and decay_summary["min_E2"] > 0.0
            ),
            "type": "physics",
            "detail": (
                "Monotonic decay="
                f"{decay_summary['vacuum_decays_monotonically']}, "
                f"max dOmegaLambda/dln a={decay_summary['max_dOmegaLambda_dln_a']:.3e}"
            ),
        },
        {
            "name": "Execution mode",
            "available": True,
            "type": "mode",
            "detail": "Prepared only; no heavy Planck/CMB chain executed by this step.",
        },
    ]


def write_yaml(info: dict[str, Any], destination: Path = YAML_PATH) -> None:
    from cobaya.yaml import yaml_dump

    destination.write_text(yaml_dump(info), encoding="utf-8")


def write_manifest(manifest: dict[str, Any], destination: Path = MANIFEST_PATH) -> None:
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_status_outputs(rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(STATUS_CSV, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [row["name"] for row in rows]
    availability = [row["available"] for row in rows]
    colors = ["#4ECDC4" if ready else "#FF6B6B" for ready in availability]
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, [1.0] * len(names), color=colors, edgecolor="black", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Dedicated DVCH Full Pipeline Readiness", fontsize=13, fontweight="bold")

    for index, (bar, row) in enumerate(zip(bars, rows, strict=False)):
        label = "Ready" if row["available"] else "Needs external setup"
        ax.text(0.5, index, label, ha="center", va="center", fontsize=8.5, fontweight="bold")

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#4ECDC4", label="Ready/configured"),
            Patch(facecolor="#FF6B6B", label="Missing runtime/data dependency"),
        ],
        loc="lower right",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(STATUS_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_readme() -> None:
    contents = """DVCH full production scaffold
============================

Purpose
- Keep the heavy Planck + Pantheon+ + BAO Cobaya/classy run separated from the lighter root-level diagnostics.
- Leave the production bundle ready to execute later without starting the expensive chain now.

Main files
- prepare_bundle.py: writes the bundle-local YAML, runtime manifest, readiness table, and example launcher.
- run_full_pipeline.py: executes the heavy Cobaya run later, once external dependencies are installed.
- dvch_full_cobaya.yaml: full Planck 2018 + Pantheon+ + BAO Cobaya configuration.
- dvch_full_runtime_manifest.json: exact runtime checklist, paths, and bridge metadata.
- status\\dvch_full_pipeline_status.csv / .png: local readiness outputs for this folder only.
- chains\\: default output directory for the future heavy chain.

External pieces still required before the real run
- Patched CLASS/classy backend with the DVCH bridge keys dvch_n, dvch_beta, dvch_model, dvch_use_exact_q.
- Cobaya packages path containing Planck 2018 clik, Pantheon+, and BAO likelihood data.

Default workflow
1. Edit run_full_pipeline_example.ps1 or pass the paths directly on the command line.
2. Run: python .\\dvch_full_pipeline\\prepare_bundle.py --packages-path <...> --classy-path <...>
3. Run: python .\\dvch_full_pipeline\\run_full_pipeline.py --packages-path <...> --classy-path <...>
4. Confirm the final chains satisfy R-1 < 0.02.

Current state
- This folder is prepared but not executed.
- The already completed lightweight and intermediate validations remain at repository root.
"""
    README_PATH.write_text(contents, encoding="utf-8")


def write_example_launcher(
    packages_path: str | None,
    classy_path: str | None,
    *,
    output_root: Path,
) -> None:
    packages_value = packages_path or r"C:\cosmo\cobaya_packages"
    classy_value = classy_path or r"C:\cosmo\class_public"
    launcher = (
        f'$PackagesPath = "{packages_value}"\n'
        f'$ClassyPath = "{classy_value}"\n'
        f'$OutputRoot = "{output_root}"\n\n'
        f'python .\\{BUNDLE_ROOT.name}\\prepare_bundle.py --packages-path $PackagesPath --classy-path $ClassyPath --output-root $OutputRoot\n'
        f'python .\\{BUNDLE_ROOT.name}\\run_full_pipeline.py --packages-path $PackagesPath --classy-path $ClassyPath --output-root $OutputRoot\n'
    )
    EXAMPLE_PS1_PATH.write_text(launcher, encoding="utf-8")
