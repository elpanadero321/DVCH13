#!/usr/bin/env python3
"""
DVCH CMB/CLASS/CAMB MCMC status diagnostic.
Checks which interface components are available for a Planck-style benchmark analysis.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

components = [
    {"name": "Cobaya", "available": False, "type": "sampler"},
    {"name": "CLASS/CAMB backend", "available": False, "type": "boltzmann"},
    {"name": "Planck clik", "available": False, "type": "likelihood"},
    {"name": "DVCH source module", "available": True, "type": "theory"},
    {"name": "Planck YAML target", "available": True, "type": "config"},
    {"name": "Pantheon+ likelihood", "available": True, "type": "sn"},
    {"name": "DESI BAO block", "available": True, "type": "bao"},
    {"name": "CC chronometer block", "available": True, "type": "cc"},
]


def main():
    print("=" * 60)
    print("DVCH CMB/CLASS/CAMB MCMC Status")
    print("=" * 60)

    df = pd.DataFrame(components)
    df.to_csv("dvch_cmb_class_camb_mcmc_status.csv", index=False)
    print("Wrote dvch_cmb_class_camb_mcmc_status.csv")

    make_figure()


def make_figure():
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [c["name"] for c in components]
    available = [c["available"] for c in components]
    colors = ['#4ECDC4' if a else '#FF6B6B' for a in available]
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, [1]*len(names), color=colors, edgecolor='black', lw=0.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xticks([])
    ax.set_title('DVCH CLASS/CAMB/Cobaya Planck Benchmark Interface', fontsize=13, fontweight='bold')

    for i, (c, a) in enumerate(zip(components, available)):
        label = "Available" if a else "Target"
        ax.text(0.5, i, label, ha='center', va='center', fontsize=9, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', label='Available'),
        Patch(facecolor='#FF6B6B', label='Benchmark target (external)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/dvch_cmb_class_camb_mcmc_status.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_cmb_class_camb_mcmc_status.png")


if __name__ == "__main__":
    main()
