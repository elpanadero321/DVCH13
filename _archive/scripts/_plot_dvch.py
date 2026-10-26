import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

PREFIX = "/mnt/d/DVCH13-1/dvch_prod"
s = loadMCSamples(PREFIX, settings={"ignore_rows": 0})

names = [p.name for p in s.getParamNames().names]
idx = {n: i for i, n in enumerate(names)}

# ============ FIG 1: corner plot DVCH (n, beta) + cosmologicos ============
g = plots.get_subplot_plotter(width_inch=8.5)
g.triangle_plot(
    s,
    ["DVCH_n", "DVCH_beta", "H0", "ns", "omch2"],
    filled=True,
    contour_colors=["#1f77b4"],
    title_limit=1,
)
g.export("/mnt/d/DVCH13-1/figures/dvch_prod_triangle.png")

# ============ FIG 2: posteriores 1D de DVCH_n y DVCH_beta vs prior ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

specs = [
    ("DVCH_n", 0.01, 0.25, "n", "#1f77b4"),
    ("DVCH_beta", 0.0, 0.001, r"$\beta$", "#d62728"),
]

for ax, (pname, lo, hi, lbl, color) in zip(axes, specs):
    x = s.samples[:, idx[pname]]
    ax.hist(x, bins=80, density=True, color=color, alpha=0.45,
            edgecolor=color, linewidth=0.8, label="posterior")
    # prior uniforme
    xs = np.linspace(lo, hi, 300)
    ax.axhline(1.0 / (hi - lo), color="k", ls="--", lw=1.2, label="prior")
    m, sd = s.mean(pname), s.std(pname)
    ax.axvline(m, color=color, lw=1.6)
    ax.axvline(m - sd, color=color, lw=0.9, ls=":")
    ax.axvline(m + sd, color=color, lw=0.9, ls=":")
    ax.set_xlabel(f"DVCH {lbl}", fontsize=12)
    ax.set_ylabel("densidad", fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_title(f"{lbl} = {m:.5g} +/- {sd:.4g}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

fig.suptitle("DVCH: posterior vs prior (Planck 2018 high-l)", fontsize=13)
fig.tight_layout()
fig.savefig("/mnt/d/DVCH13-1/figures/dvch_posterior_vs_prior.png", dpi=140)
plt.close(fig)

# ============ FIG 3: trazas de las cadenas (DVCH_n, DVCH_beta) ============
offs = list(s.chain_offsets)
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for k in range(len(offs) - 1):
    sl = slice(offs[k], offs[k + 1])
    axes[0].plot(s.samples[sl, idx["DVCH_n"]], lw=0.35, alpha=0.75, label=f"cadena {k+1}")
    axes[1].plot(s.samples[sl, idx["DVCH_beta"]], lw=0.35, alpha=0.75)
axes[0].set_ylabel("DVCH_n", fontsize=12)
axes[1].set_ylabel(r"DVCH_$\beta$", fontsize=12)
axes[1].set_xlabel("muestra (concatenada)", fontsize=11)
axes[0].legend(fontsize=8, loc="upper right", ncol=4)
axes[0].set_title("Trazas MCMC por cadena", fontsize=13)
fig.tight_layout()
fig.savefig("/mnt/d/DVCH13-1/figures/dvch_traces.png", dpi=140)
plt.close(fig)

print("FIGURAS GENERADAS:")
print("  figures/dvch_prod_triangle.png")
print("  figures/dvch_posterior_vs_prior.png")
print("  figures/dvch_traces.png")
