#!/usr/bin/env python3
"""
Run the dedicated DVCH full-production Cobaya pipeline once the heavy runtime
dependencies are installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bundle_common import (
    DEFAULT_OUTPUT_ROOT,
    DVCHParameters,
    build_full_info,
    build_runtime_manifest,
    build_status_rows,
    ensure_layout,
    fiducial_decay_summary,
    module_available,
    write_example_launcher,
    write_manifest,
    write_readme,
    write_status_outputs,
    write_yaml,
)


def check_gelman_rubin(output_root: str, limit: float) -> float:
    from getdist.mcsamples import loadMCSamples

    samples = loadMCSamples(output_root, no_cache=True)
    chains = samples.getSeparateChains()
    if len(chains) < 2:
        raise RuntimeError(
            "Gelman-Rubin requires at least two separate chains. Run Cobaya with multiple chains."
        )
    rminus1 = float(samples.getGelmanRubin(chainlist=chains))
    if not np.isfinite(rminus1):
        raise RuntimeError("GetDist returned a non-finite Gelman-Rubin statistic.")
    if rminus1 >= limit:
        raise RuntimeError(f"Convergence not reached: R-1 = {rminus1:.5f} >= {limit:.2f}.")
    return rminus1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the separate DVCH full-production Cobaya pipeline."
    )
    parser.add_argument(
        "--packages-path",
        required=True,
        help="Cobaya packages directory containing Planck/SN/BAO data files.",
    )
    parser.add_argument(
        "--classy-path",
        required=True,
        help="Path to the patched CLASS source tree used by Cobaya's classy wrapper.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root path for the full Cobaya chain output.",
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
        "--debug",
        action="store_true",
        help="Enable Cobaya debug logging during the run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing Cobaya output root instead of forcing a fresh run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwriting an existing Cobaya output root for a fresh run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    packages_path = Path(args.packages_path)
    classy_path = Path(args.classy_path)

    if not packages_path.exists():
        raise RuntimeError(f"Packages path does not exist: {packages_path}")
    if not classy_path.exists():
        raise RuntimeError(f"Patched classy path does not exist: {classy_path}")
    if not module_available("cobaya"):
        raise RuntimeError("Cobaya is not importable in the current Python environment.")
    if not module_available("getdist"):
        raise RuntimeError("GetDist is not importable in the current Python environment.")

    ensure_layout(output_root)
    info = build_full_info(
        classy_path=str(classy_path),
        output_root=output_root,
        gr_limit=args.gr_limit,
        max_samples=args.max_samples,
    )
    decay_summary = fiducial_decay_summary(DVCHParameters())

    write_yaml(info)
    write_readme()
    write_example_launcher(str(packages_path), str(classy_path), output_root=output_root)
    write_manifest(
        build_runtime_manifest(
            packages_path=str(packages_path),
            classy_path=str(classy_path),
            output_root=output_root,
            gr_limit=args.gr_limit,
            max_samples=args.max_samples,
            info=info,
            decay_summary=decay_summary,
            execution_state="run_requested",
        )
    )
    write_status_outputs(
        build_status_rows(
            packages_path=str(packages_path),
            classy_path=str(classy_path),
            info=info,
            decay_summary=decay_summary,
            yaml_written=True,
            manifest_written=True,
        )
    )

    from cobaya.run import run

    run(
        info,
        packages_path=str(packages_path),
        output=str(output_root),
        debug=args.debug,
        stop_at_error=True,
        resume=args.resume,
        force=args.force,
    )
    rminus1 = check_gelman_rubin(str(output_root), args.gr_limit)
    write_manifest(
        build_runtime_manifest(
            packages_path=str(packages_path),
            classy_path=str(classy_path),
            output_root=output_root,
            gr_limit=args.gr_limit,
            max_samples=args.max_samples,
            info=info,
            decay_summary=decay_summary,
            execution_state="completed",
            gelman_rubin_rminus1=rminus1,
        )
    )

    print("=" * 72)
    print("Dedicated DVCH full-production bundle")
    print("=" * 72)
    print(f"Completed Cobaya run with R-1 = {rminus1:.5f}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
