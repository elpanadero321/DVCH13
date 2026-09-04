"""Compute basic independent-chain diagnostics for Cobaya text outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def gelman_rubin(chains: list[np.ndarray]) -> np.ndarray:
    n = min(len(chain) for chain in chains)
    trimmed = np.asarray([chain[:n] for chain in chains], dtype=float)
    m = trimmed.shape[0]
    means = trimmed.mean(axis=1)
    within = trimmed.var(axis=1, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    return np.sqrt(((n - 1) * within + between) / (n * within))


def effective_sample_size(chains: list[np.ndarray]) -> np.ndarray:
    merged = np.concatenate(chains, axis=0)
    ess = np.empty(merged.shape[1])
    for index in range(merged.shape[1]):
        values = merged[:, index] - merged[:, index].mean()
        variance = np.var(values)
        if variance == 0:
            ess[index] = float(len(values))
            continue
        rho_sum = 0.0
        for lag in range(1, min(len(values) // 2, 1000)):
            rho = np.dot(values[:-lag], values[lag:]) / (
                (len(values) - lag) * variance
            )
            if rho <= 0:
                break
            rho_sum += rho
        ess[index] = len(values) / (1.0 + 2.0 * rho_sum)
    return ess


def load(path: Path, columns: list[str]) -> np.ndarray:
    header = path.read_text().splitlines()[0].lstrip("#").split()
    frame = pd.read_csv(path, sep=r"\s+", names=header, skiprows=1)
    return frame[columns].to_numpy(dtype=float)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python dvch_planck_chain_diagnostics.py CHAIN_A CHAIN_B")
    first_path = Path(sys.argv[1])
    header = first_path.read_text().splitlines()[0].lstrip("#").split()
    first = pd.DataFrame(columns=header)
    columns = [
        column for column in first.columns
        if column not in {"weight", "minuslogpost", "minuslogprior", "chi2"}
        and not column.startswith("minuslogprior__")
        and not column.startswith("chi2__")
    ]
    chains = [load(Path(path), columns) for path in sys.argv[1:]]
    rhat = gelman_rubin(chains)
    ess = effective_sample_size(chains)
    result = pd.DataFrame({"parameter": columns, "Rhat": rhat, "ESS": ess})
    output = Path("dvch_planck_chain_diagnostics.csv")
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
