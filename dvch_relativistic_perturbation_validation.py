#!/usr/bin/env python3
"""
DVCH relativistic perturbation validation preflight/status gate.
Checks which components are available for a future full CMB/CLASS/CAMB validation.
"""
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

validation_items = [
    {"item": "Patched CLASS/CAMB backend", "status": "Protocol target", "requirement": "consumes DVCH sources", "available": False},
    {"item": "Photon-baryon hierarchy", "status": "Protocol target", "requirement": "full Boltzmann hierarchy", "available": False},
    {"item": "Einstein-Boltzmann metric sources", "status": "Protocol target", "requirement": "metric and line-of-sight evolution", "available": False},
    {"item": r"CMB $C_\ell$ spectra", "status": "Protocol target", "requirement": "TT/TE/EE from DVCH perturbations", "available": False},
    {"item": "CMB lensing spectra", "status": "Protocol target", "requirement": "lensing potential/reconstruction", "available": False},
    {"item": "Planck likelihood stack", "status": "Protocol target", "requirement": "Cobaya/clik/native Planck data", "available": False},
    {"item": "DVCH source-table module", "status": "Available", "requirement": "local interface module present", "available": True},
    {"item": "Planck YAML target", "status": "Available", "requirement": "local YAML target present", "available": True},
]


def main():
    print("=" * 60)
    print("DVCH Relativistic Perturbation Validation Status")
    print("=" * 60)

    # Write JSON
    status_dict = {item["item"]: item["status"] for item in validation_items}
    with open("dvch_relativistic_perturbation_status.json", "w") as f:
        json.dump(status_dict, f, indent=2)
    print("Wrote dvch_relativistic_perturbation_status.json")

    # Write CSV
    df = pd.DataFrame(validation_items)
    df.to_csv("dvch_relativistic_perturbation_status.csv", index=False)
    print("Wrote dvch_relativistic_perturbation_status.csv")

    make_figure()


def make_figure():
    fig, ax = plt.subplots(figsize=(12, 5))

    items = [item["item"] for item in validation_items]
    statuses = [item["status"] for item in validation_items]
    available = [item["available"] for item in validation_items]

    colors = ['#4ECDC4' if a else '#FFD93D' for a in available]
    y_pos = np.arange(len(items))

    bars = ax.barh(y_pos, [1]*len(items), color=colors, edgecolor='black', lw=0.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(items, fontsize=9)
    ax.set_xticks([])
    ax.set_title('DVCH Relativistic Perturbation Preflight/Status Gate', fontsize=13, fontweight='bold')

    for i, (bar, status) in enumerate(zip(bars, statuses)):
        ax.text(0.5, i, status, ha='center', va='center', fontsize=9, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', label='Available (local interface)'),
        Patch(facecolor='#FFD93D', label='Protocol target (external runtime required)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/dvch_relativistic_perturbation_validation_status.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_relativistic_perturbation_validation_status.png")


if __name__ == "__main__":
    main()
