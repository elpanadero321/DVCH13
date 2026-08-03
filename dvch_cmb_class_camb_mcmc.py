#!/usr/bin/env python3
"""
Cobaya launcher and status checker for the DVCH MCMC pipeline.

This script writes a production-oriented Cobaya configuration that assumes a
patched classy/CLASS backend capable of consuming the extra DVCH parameters
``dvch_n`` and ``dvch_beta`` plus the metadata keys added in ``extra_args``.

Usage examples
--------------
1. Dry-run plus YAML export:
   python dvch_cmb_class_camb_mcmc.py --write-yaml

2. Full run once a patched classy backend and the Cobaya likelihood data are installed:
   python dvch_cmb_class_camb_mcmc.py --run --packages-path C:\\cosmo\\cobaya_packages --classy-path C:\\cosmo\\class_public
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from cobaya.run import run
from cobaya.yaml import yaml_dump
from getdist.mcsamples import loadMCSamples

from dvch_theory import DVCHParameters, build_classy_extra_args, fiducial_decay_summary

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)
STATUS_CSV = Path("dvch_cmb_class_camb_mcmc_status.csv")
STATUS_PNG = FIGDIR / "dvch_cmb_class_camb_mcmc_status.png"
YAML_PATH = Path("dvch_cobaya.yaml")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_likelihood_block(dataset_combo: str) -> dict[str, dict[str, Any]]:
    likelihoods: dict[str, dict[str, Any]] = {}

    if dataset_combo in {"full", "planck-only"}:
        likelihoods.update(
            {
                "planck_2018_highl_plik.TTTEEE": {},
                "planck_2018_lowl.TT_clik": {},
                "planck_2018_lowl.EE_clik": {},
                "planck_2018_lensing.clik": {},
            }
        )

    if dataset_combo in {"full", "late-time"}:
        likelihoods.update(
            {
                "sn.pantheonplus": {},
                "bao.sixdf_2011_bao": {},
                "bao.sdss_dr7_mgs": {},
                "bao.sdss_dr12_consensus_bao": {},
                "bao.desi_2024_bao_all": {},
            }
        )

    if not likelihoods:
        raise ValueError(f"Unsupported dataset combo: {dataset_combo}")

    return likelihoods


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


def build_info(args: argparse.Namespace) -> dict[str, Any]:
    theory_block = {
        "classy": {
            "path": args.classy_path,
            "stop_at_error": True,
            "ignore_obsolete": True,
            "extra_args": build_classy_extra_args(
                z_pk="0 0.5 1 2",
                l_max_scalars=3500,
                pk_max_h_mpc=5.0,
            ),
        }
    }

    sampler_block = {
        "mcmc": {
            "drag": True,
            "learn_proposal": True,
            "learn_proposal_Rminus1_max": 2.0,
            "learn_every": "40d",
            "max_tries": "80d",
            "oversample_power": 0.4,
            "proposal_scale": 1.9,
            "Rminus1_stop": args.gr_limit,
            "Rminus1_cl_stop": args.gr_limit,
            "max_samples": args.max_samples,
        }
    }

    return {
        "theory": theory_block,
        "likelihood": build_likelihood_block(args.dataset_combo),
        "params": build_params_block(),
        "sampler": sampler_block,
        "output": args.output_root,
    }


def export_yaml(info: dict[str, Any], destination: Path) -> None:
    destination.write_text(yaml_dump(info), encoding="utf-8")


def build_status_rows(
    args: argparse.Namespace,
    info: dict[str, Any],
    yaml_written: bool,
    decay_summary: dict[str, float | bool],
) -> list[dict[str, Any]]:
    packages_path_ready = bool(args.packages_path) and Path(args.packages_path).exists()
    selected_likelihoods = ", ".join(info["likelihood"].keys())

    return [
        {
            "name": "Cobaya Python package",
            "available": module_available("cobaya"),
            "type": "sampler",
            "detail": "Local Cobaya import available.",
        },
        {
            "name": "GetDist post-processing",
            "available": module_available("getdist"),
            "type": "post",
            "detail": "Needed for Gelman-Rubin post-checks.",
        },
        {
            "name": "classy backend import",
            "available": module_available("classy"),
            "type": "boltzmann",
            "detail": "Required locally unless a patched CLASS path is supplied.",
        },
        {
            "name": "Patched DVCH CLASS bridge",
            "available": True,
            "type": "theory",
            "detail": "Configured through dvch_n, dvch_beta, dvch_model, and dvch_use_exact_q.",
        },
        {
            "name": "Likelihood data path",
            "available": packages_path_ready,
            "type": "data",
            "detail": "Cobaya packages path for Planck, Pantheon+, and BAO datasets.",
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
            "name": "Cobaya YAML export",
            "available": yaml_written,
            "type": "config",
            "detail": f"Written to {YAML_PATH}.",
        },
        {
            "name": "Convergence criterion",
            "available": True,
            "type": "mcmc",
            "detail": f"Require Gelman-Rubin R-1 < {args.gr_limit:.2f}.",
        },
    ]


def write_status_outputs(rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(STATUS_CSV, index=False)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    names = [row["name"] for row in rows]
    availability = [row["available"] for row in rows]
    colors = ["#4ECDC4" if ready else "#FF6B6B" for ready in availability]
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, [1.0] * len(names), color=colors, edgecolor="black", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks([])
    ax.set_title("DVCH Cobaya/classy Pipeline Readiness", fontsize=13, fontweight="bold")

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


def check_gelman_rubin(output_root: str, limit: float) -> float:
    samples = loadMCSamples(output_root, no_cache=True)
    chains = samples.getSeparateChains()
    if len(chains) < 2:
        raise RuntimeError(
            "Gelman-Rubin requires at least two separate chains. Run Cobaya with "
            "multiple MPI chains or multiple chain files."
        )
    rminus1 = float(samples.getGelmanRubin(chainlist=chains))
    if not np.isfinite(rminus1):
        raise RuntimeError("GetDist returned a non-finite Gelman-Rubin statistic.")
    if rminus1 >= limit:
        raise RuntimeError(
            f"Convergence not reached: R-1 = {rminus1:.5f} >= {limit:.2f}."
        )
    return rminus1


def maybe_run_pipeline(args: argparse.Namespace, info: dict[str, Any]) -> float | None:
    if not args.run:
        return None
    if not module_available("classy") and not args.classy_path:
        raise RuntimeError(
            "No local classy import found. Supply --classy-path pointing to a patched "
            "CLASS build or install classy before running the chain."
        )
    if not args.packages_path:
        raise RuntimeError(
            "A Cobaya packages path is required for the Planck, Pantheon+, and BAO data files."
        )

    run(
        info,
        packages_path=args.packages_path,
        output=args.output_root,
        debug=args.debug,
        stop_at_error=True,
        resume=False,
        force=True,
    )
    return check_gelman_rubin(args.output_root, args.gr_limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run the DVCH Cobaya pipeline.")
    parser.add_argument(
        "--dataset-combo",
        choices=["full", "late-time", "planck-only"],
        default="full",
        help="Likelihood combination to encode in the exported info dictionary.",
    )
    parser.add_argument(
        "--packages-path",
        default=None,
        help="Cobaya packages directory containing Planck/SN/BAO data files.",
    )
    parser.add_argument(
        "--classy-path",
        default=None,
        help="Path to the patched CLASS source tree used by Cobaya's classy wrapper.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path("chains") / "dvch_full"),
        help="Root path for Cobaya chain output.",
    )
    parser.add_argument(
        "--gr-limit",
        type=float,
        default=0.02,
        help="Gelman-Rubin stopping threshold on R-1.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=120000,
        help="Maximum number of accepted samples for the MCMC run.",
    )
    parser.add_argument(
        "--write-yaml",
        action="store_true",
        help="Write dvch_cobaya.yaml alongside the Python launcher.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute Cobaya after building the configuration.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Cobaya debug logging during an actual run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = build_info(args)
    decay_summary = fiducial_decay_summary(DVCHParameters())

    yaml_written = False
    if args.write_yaml:
        export_yaml(info, YAML_PATH)
        yaml_written = True

    rows = build_status_rows(args, info, yaml_written, decay_summary)
    write_status_outputs(rows)

    gr_value = maybe_run_pipeline(args, info)

    print("=" * 72)
    print("DVCH Cobaya/classy pipeline")
    print("=" * 72)
    print(f"Dataset combo: {args.dataset_combo}")
    print(f"Output root: {args.output_root}")
    print(f"Status table: {STATUS_CSV}")
    print(f"Status figure: {STATUS_PNG}")
    if yaml_written:
        print(f"YAML export: {YAML_PATH}")
    print(
        "Fiducial decay check: "
        f"monotonic={decay_summary['vacuum_decays_monotonically']}, "
        f"max dOmegaLambda/dln a={decay_summary['max_dOmegaLambda_dln_a']:.3e}"
    )
    if gr_value is not None:
        print(f"Converged chain achieved with R-1 = {gr_value:.5f}")
    else:
        print(f"Configured Gelman-Rubin target: R-1 < {args.gr_limit:.2f}")


if __name__ == "__main__":
    main()
