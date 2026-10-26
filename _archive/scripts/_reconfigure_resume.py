#!/usr/bin/env python3
"""Reconfigure the stored run info for the definitive resumed phase.

Cobaya restores these keys from the OLD run info on resume
(_at_resume_prefer_old): max_samples, Rminus1_stop, covmat, learn_every, ...
So to change them we must edit the stored info (dill pickle preferred by
reload_updated_info, plus the yaml for consistency), not just the env vars.

Changes:
  - Rminus1_stop:  0.01 -> 0.03   (chosen budget: ESS ~1400/param, ~9 days)
  - max_samples:   20000 -> 40000 (safety ceiling; R-1 stop triggers first)
  - learn_proposal: true -> false (freeze the verified-optimal proposal ->
    clean, time-homogeneous Markov chain for the final sample)
"""
import shutil
from pathlib import Path

import dill

ROOT = Path("/mnt/d/DVCH13-1")
PREFIX = "dvch_prod"
BACKUP = ROOT / "dvch_backup_phase3"

FILES = [
    f"{PREFIX}.1.txt", f"{PREFIX}.2.txt", f"{PREFIX}.3.txt", f"{PREFIX}.4.txt",
    f"{PREFIX}.checkpoint", f"{PREFIX}.checkpoint.bak", f"{PREFIX}.covmat",
    f"{PREFIX}.progress", f"{PREFIX}.input.yaml", f"{PREFIX}.input.yaml.locked",
    f"{PREFIX}.updated.yaml", f"{PREFIX}.updated.dill_pickle",
    "dvch_prod_run.log",
]

NEW = {
    "Rminus1_stop": 0.03,
    "max_samples": 40000,
    "learn_proposal": False,
}


def backup():
    BACKUP.mkdir(exist_ok=True)
    for f in FILES:
        src = ROOT / f
        if src.is_file():
            shutil.copy2(src, BACKUP / f)
    print(f"backed up {len(list(BACKUP.iterdir()))} files to {BACKUP}")


def edit_pickle():
    p = ROOT / f"{PREFIX}.updated.dill_pickle"
    with open(p, "rb") as fh:
        info = dill.load(fh)
    mcmc = info["sampler"]["mcmc"]
    print("before:", {k: mcmc.get(k) for k in NEW})
    mcmc.update(NEW)
    print("after :", {k: mcmc.get(k) for k in NEW})
    with open(p, "wb") as fh:
        dill.dump(info, fh)
    print(f"patched {p}")


def edit_yaml():
    """Text-level edit so the yaml and pickle agree."""
    p = ROOT / f"{PREFIX}.updated.yaml"
    text = p.read_text()
    repl = {
        "Rminus1_stop: 0.01": "Rminus1_stop: 0.03",
        "max_samples: 20000": "max_samples: 40000",
        "learn_proposal: true": "learn_proposal: false",
    }
    for old, new in repl.items():
        assert old in text, f"pattern not found: {old}"
        text = text.replace(old, new)
    p.write_text(text)
    print(f"patched {p}")


if __name__ == "__main__":
    backup()
    edit_pickle()
    edit_yaml()
