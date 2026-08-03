#!/usr/bin/env python3
"""
Prepare the dedicated DVCH full-production folder without running the heavy chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bundle_common import (
    DEFAULT_OUTPUT_ROOT,
    DVCHParameters,
    build_full_info,
    build_runtime_manifest,
    build_status_rows,
    ensure_layout,
    fiducial_decay_summary,
    write_example_launcher,
    write_manifest,
    write_readme,
    write_status_outputs,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the separate DVCH full-production bundle without running Cobaya."
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
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root path for the future full Cobaya chain output.",
    )
    parser.add_argument(
        "--gr-limit",
        type=float,
        default=0.02,
        help="Gelman-Rubin stopping threshold on R-1 for the future run.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=120000,
        help="Maximum number of accepted samples for the future full run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_layout(output_root)

    info = build_full_info(
        classy_path=args.classy_path,
        output_root=output_root,
        gr_limit=args.gr_limit,
        max_samples=args.max_samples,
    )
    decay_summary = fiducial_decay_summary(DVCHParameters())

    write_yaml(info)
    manifest = build_runtime_manifest(
        packages_path=args.packages_path,
        classy_path=args.classy_path,
        output_root=output_root,
        gr_limit=args.gr_limit,
        max_samples=args.max_samples,
        info=info,
        decay_summary=decay_summary,
        execution_state="prepared_not_run",
    )
    write_manifest(manifest)
    write_readme()
    write_example_launcher(args.packages_path, args.classy_path, output_root=output_root)

    rows = build_status_rows(
        packages_path=args.packages_path,
        classy_path=args.classy_path,
        info=info,
        decay_summary=decay_summary,
        yaml_written=True,
        manifest_written=True,
    )
    write_status_outputs(rows)

    print("=" * 72)
    print("Dedicated DVCH full-production bundle")
    print("=" * 72)
    print("State: prepared only (no heavy Cobaya chain executed)")
    print(f"Output root: {output_root}")
    print("Files written:")
    print("  - dvch_full_pipeline\\dvch_full_cobaya.yaml")
    print("  - dvch_full_pipeline\\dvch_full_runtime_manifest.json")
    print("  - dvch_full_pipeline\\README.txt")
    print("  - dvch_full_pipeline\\run_full_pipeline_example.ps1")
    print("  - dvch_full_pipeline\\status\\dvch_full_pipeline_status.csv")
    print("  - dvch_full_pipeline\\status\\dvch_full_pipeline_status.png")


if __name__ == "__main__":
    main()
