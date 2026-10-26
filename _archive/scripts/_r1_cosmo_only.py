"""R-1 restricted to the 8 cosmological parameters of the existing chains.

The convergence criterion that matters physically is the one on the sampled
cosmology. The nuisance parameters are amplitude/calibration bookkeeping; their
mixing should not gate the scientific result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _r1_diagnostic import cobaya_rminus1, classic_gr, load, sampled_cols, header_names

COSMO = ("H0", "ombh2", "omch2", "tau", "ln10As", "ns", "DVCH_n", "DVCH_beta")


def main(prefix: str) -> None:
    root = Path(__file__).resolve().parent
    files = sorted(root.glob(f"{prefix}.*.txt"))
    names = header_names(files[0])
    chains_all = [sampled_cols(load(f), names) for f in files]
    col_names = [n for n in names if n not in (
        "weight", "minuslogpost") and not n.startswith("chi2")
        and not n.startswith("minuslogprior")]
    cidx = [col_names.index(c) for c in COSMO]

    def report(tag: str, chains):
        for frac in (0.0, 0.5):
            cut = [c[int(len(c) * frac):] for c in chains]
            r1, _, _ = cobaya_rminus1(cut, 0.5)
            rg = classic_gr(cut)
            print(f"  {tag:>18} cut={frac:.0%}  cobaya-R1={r1:.4f}  "
                  f"classic max R-1={np.nanmax(rg):.4f}")

    print(f"=== {prefix}: {len(files)} chains, "
          f"{[len(c) for c in chains_all]} samples ===")
    print("\nALL sampled params (26):")
    report("all 26", chains_all)
    print("\n8 cosmological params only:")
    report("cosmo 8", [c[:, cidx] for c in chains_all])
    print("\nnuisance only (18):")
    nidx = [i for i, n in enumerate(col_names) if n not in COSMO]
    report("nuisance 18", [c[:, nidx] for c in chains_all])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dvch_prod")
