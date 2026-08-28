#!/usr/bin/env python3
"""
DVCH Planck preflight checker.

Verifies the local environment for running a full Planck 2018 CMB
likelihood analysis with DVCH. Reports which components are available
and which are missing, so the user knows what must be installed before
launching the external Cobaya pipeline.

The preflight is a non-invasive, read-only environment check; it does
not download, install, or modify anything.
"""
import importlib.util as _u
import sys
import os
import json

RESULTS = {}


def _check(mod, name=None):
    if name is None:
        name = mod
    spec = _u.find_spec(mod)
    available = bool(spec)
    RESULTS[name] = available
    return available


def _check_file(path):
    exists = os.path.isfile(path)
    RESULTS[path] = exists
    return exists


def _check_env(var):
    has = bool(os.environ.get(var))
    RESULTS[f"env:{var}"] = has
    return has


def main():
    print("=== DVCH Planck preflight ===")
    print()

    # Python version
    print(f"Python  : {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    # Core Python packages
    print("--- Core packages ---")
    _check("numpy")
    _check("scipy")
    _check("matplotlib")
    _check("pandas")

    # MCMC / sampling
    print("\n--- MCMC / sampling ---")
    _check("emcee")
    _check("corner")
    _check("getdist")
    _check("cobaya")

    # Boltzmann solvers
    print("\n--- Boltzmann solvers ---")
    _check("classy", "classy (CLASS)")
    _check("camb", "camb (CAMB)")

    # Planck likelihood
    print("\n--- Planck likelihoods ---")
    _check("clik")
    _check("plancklens")

    # MPI
    _check("mpi4py")

    # DVCH local files
    print("\n--- DVCH local files ---")
    _check_file("dvch_planck2018_full.yaml")
    _check_file("dvch_boltzmann_backend.py")
    _check_file("dvch_mcmc_chains_full.csv")
    _check_file("dvch_full_mcmc_pipeline.py")

    # Planck data directory (typical paths)
    plk = os.environ.get("PLANCK_DATA_DIR", "")
    print(f"\nPLANCK_DATA_DIR = {plk if plk else '(not set)'}")

    # Summary
    print("\n=== Summary ===")
    status = {k: "OK" if v else "MISSING" for k, v in RESULTS.items()}
    for name, st in sorted(status.items()):
        print(f"  {name:<30s} {st}")

    # Write JSON report
    with open("dvch_planck_preflight_report.json", "w") as fh:
        json.dump({"status": status, "timestamp": __import__("datetime").datetime.now().isoformat()}, fh, indent=2)
    print("\nReport written to dvch_planck_preflight_report.json")


if __name__ == "__main__":
    main()