"""Compute independent-chain convergence diagnostics for Cobaya text outputs.

Handles the repeated-weight convention used by Cobaya's text output: each row
carries a `weight` column (integer >= 1) counting how many times that sample
was accepted.  R-hat (Gelman-Rubin) and ESS are computed on the weight-expanded
chains so that repeated samples are treated correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that are not sampled parameters and must be excluded from diagnostics.
_NON_PARAM = {
    "weight",
    "minuslogpost",
    "minuslogprior",
    "chi2",
}


def _is_param(column: str) -> bool:
    if column in _NON_PARAM:
        return False
    if column.startswith("minuslogprior__") or column.startswith("chi2__"):
        return False
    return True


def load(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (values, weights, parameter_names) for one chain file."""
    header = path.read_text().splitlines()[0].lstrip("#").split()
    frame = pd.read_csv(path, sep=r"\s+", names=header, skiprows=1)
    columns = [column for column in header if _is_param(column)]
    values = frame[columns].to_numpy(dtype=float)
    weights = frame["weight"].to_numpy(dtype=int) if "weight" in frame else None
    return values, weights, columns


def _expand(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Expand a chain by its integer weights (repeated samples)."""
    if weights is None:
        return values
    return np.repeat(values, weights, axis=0)


def gelman_rubin(chains: list[np.ndarray]) -> np.ndarray:
    """Gelman-Rubin R-hat across chains (weight-expanded, equal length)."""
    n = min(len(chain) for chain in chains)
    trimmed = np.asarray([chain[:n] for chain in chains], dtype=float)
    m = trimmed.shape[0]
    means = trimmed.mean(axis=1)
    within = trimmed.var(axis=1, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(((n - 1) * within + between) / (n * within))
    # A parameter with zero within-chain variance is perfectly converged.
    rhat = np.where(np.isfinite(rhat), rhat, 1.0)
    return rhat


def effective_sample_size(chains: list[np.ndarray]) -> np.ndarray:
    """Effective sample size per parameter, pooled across chains."""
    merged = np.concatenate(chains, axis=0)
    n_total = merged.shape[0]
    ess = np.empty(merged.shape[1])
    for index in range(merged.shape[1]):
        values = merged[:, index] - merged[:, index].mean()
        variance = np.var(values)
        if variance == 0:
            ess[index] = float(n_total)
            continue
        rho_sum = 0.0
        max_lag = min(n_total // 2, 1000)
        for lag in range(1, max_lag):
            rho = np.dot(values[:-lag], values[lag:]) / (
                (n_total - lag) * variance
            )
            if rho <= 0:
                break
            rho_sum += rho
        ess[index] = n_total / (1.0 + 2.0 * rho_sum)
    return ess


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python dvch_planck_chain_diagnostics.py CHAIN_A CHAIN_B [CHAIN_C ...]"
        )
    loaded = [load(Path(path)) for path in sys.argv[1:]]
    columns = loaded[0][2]
    for _, _, cols in loaded[1:]:
        if cols != columns:
            raise SystemExit("chains have mismatched parameter columns")
    chains = [_expand(values, weights) for values, weights, _ in loaded]
    rhat = gelman_rubin(chains)
    ess = effective_sample_size(chains)
    n_total = sum(len(chain) for chain in chains)
    result = pd.DataFrame(
        {
            "parameter": columns,
            "Rhat": rhat,
            "ESS": ess,
            "ESS_per_sample": ess / n_total,
        }
    )
    output = Path("dvch_planck_chain_diagnostics.csv")
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Total weighted samples across {len(chains)} chains: {n_total}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
