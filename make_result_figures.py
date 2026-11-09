#!/usr/bin/env python3
# Genera las figuras del paper a partir de los resultados del analisis
# tardio con datos reales:
#  (1) corner plot: posteriors late-time LCDM vs DVCH
#  (2) comparacion de modelos: log Z (LCDM, w0waCDM, DVCH)
#  (3) residuales (CC, DESI BAO, Pantheon+) del mejor ajuste de cada modelo
# Lee dvch_real_chain_*.npy, dvch_late_time_fit_summary.csv y
# dvch_evidence_summary.csv producidos por dvch_vs_lcdm_real.py y
# bayes_factor_real.py.
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _load_summary():
    path = os.path.join(BASE, "dvch_late_time_fit_summary.csv")
    df = pd.read_csv(path)
    out = {}
    for model in ("lcdm", "dvch"):
        sub = df[df.model == model]
        params = {}
        for _, row in sub.iterrows():
            for key in row.index:
                if key.endswith("_median") and np.isfinite(row[key]):
                    params[key[:-7]] = float(row[key])
        out[model] = {"chi2_min": float(sub["chi2_min"].iloc[0]),
                      "params": params}
    return out


# ---------------- (1) corner plot LCDM vs DVCH ----------------
lcdm = np.load(os.path.join(BASE, "dvch_real_chain_lcdm.npy"))   # Om,H0,M,rd
dvch = np.load(os.path.join(BASE, "dvch_real_chain_dvch.npy"))   # Om,H0,M,n,beta,rd

labels = [r"$\Omega_m$", r"$H_0$", r"$M$", r"$n$", r"$\beta$"]
ranges = [(0.22, 0.40), (65.5, 71.5), (-19.75, -19.05), (-0.1, 1.0), (-0.2, 2.2)]

fig, axes = plt.subplots(5, 5, figsize=(11, 10))
for i in range(5):
    for j in range(5):
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        if i >= 3 and j < 3:
            ax.axis("off")  # LCDM no tiene (n, beta)
            continue
        if i == j:
            if i < 3:  # (n, beta) solo existen en DVCH
                ax.hist(lcdm[:, i], bins=60, density=True, alpha=0.5,
                        color="C0")
            ax.hist(dvch[:, i], bins=60, density=True, alpha=0.5, color="C1")
            ax.set_xlim(*ranges[i])
        else:
            if i < 3:
                x, y = lcdm[:, j], lcdm[:, i]
            else:
                x, y = dvch[:, j], dvch[:, i]
            ax.plot(x, y, ",", alpha=0.02, color="C1")
            ax.set_xlim(*ranges[j])
            ax.set_ylim(*ranges[i])
        if j > 0:
            ax.set_yticks([])
        if i < 4:
            ax.set_xticks([])
    if i == 4:
        for j in range(5):
            axes[i, j].set_xlabel(labels[j], fontsize=11)
    axes[i, 0].set_ylabel(labels[i], fontsize=11)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="C0", label=r"$\Lambda$CDM"),
                    Patch(color="C1", label="DVCH")],
           loc="upper right", fontsize=12)
fig.suptitle(r"Late-time posteriors (Pantheon$+$ 1580 SNe (z>0.01, sin calibradoras)"
             r" + DESI DR1 BAO + CC, real data)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIGDIR, "dvch_latetime_corner.png"), dpi=150,
            bbox_inches="tight")
print("corner plot guardado")

# ---------------- (2) comparacion de modelos ----------------
ev = pd.read_csv(os.path.join(BASE, "dvch_evidence_summary.csv"))
logz = ev.logZ.tolist()
errs = ev.logZ_err.tolist()
names = [r"$\Lambda$CDM", r"$w_0w_a$CDM", "DVCH"]

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ["C0", "C2", "C3"]
bars = ax.bar(names, logz, yerr=errs, color=colors, capsize=6,
              edgecolor="k", linewidth=0.6)
for b, v in zip(bars, logz):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}",
            ha="center", fontsize=11)
ax.axhline(logz[2], ls="--", lw=1, color="C3", alpha=0.5)
ax.annotate(r"$\Delta\ln Z$(DVCH$-\Lambda$CDM) $=%+.2f$" % (logz[2] - logz[0]),
            xy=(1, logz[2] + 0.5), fontsize=11, color="C3")
ax.set_ylabel(r"$\ln Z$ (nested sampling)", fontsize=12)
ax.set_title(r"Bayesian evidence on Pantheon$+$ + DESI DR1 BAO + CC (real data)",
             fontsize=13)
ax.set_ylim(min(logz) - 8, max(logz) + 5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "dvch_model_comparison.png"), dpi=150,
            bbox_inches="tight")
print("model comparison guardado")

# ---------------- (3) residuales del mejor ajuste ----------------
import dvch_vs_lcdm_real as m

sne = m.load_pantheon()
z_hd, z_hel, mb, dmb, cov, cinv = sne
summ = _load_summary()
lcdm_p = summ["lcdm"]["params"]
dvch_p = summ["dvch"]["params"]


def model_p(model, key):
    return (lcdm_p if model == "lcdm" else dvch_p)[key]


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# CC
ax1 = axes[0]
cc_z = m.cc_data[:, 0]
cc_H = m.cc_data[:, 1]
cc_err = m.cc_data[:, 2]
for model, color, marker in (("lcdm", "r", "o"), ("dvch", "b", "s")):
    e2 = m.E2_lcdm(cc_z, model_p(model, "Om")) if model == "lcdm" else \
        m.E2_dvch(cc_z, model_p(model, "Om"), model_p(model, "n"),
                  model_p(model, "beta"))
    H = model_p(model, "H0") * np.sqrt(e2)
    ax1.errorbar(cc_z + (0.008 if model == "dvch" else -0.008),
                 (cc_H - H) / cc_err, yerr=1, fmt=marker, ms=4, capsize=2,
                 color=color, label=("DVCH" if model == "dvch" else r"$\Lambda$CDM"),
                 alpha=0.8)
ax1.axhline(0, color="k", ls="--", lw=0.5)
ax1.set_xlabel("z")
ax1.set_ylabel(r"$\Delta H / \sigma$")
ax1.set_title(f"Cosmic Chronometer Residuals ({len(cc_z)} puntos)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# BAO
ax2 = axes[1]
for model, color, marker in (("lcdm", "r", "o"), ("dvch", "b", "s")):
    funcs = m.model_functions(
        model, model_p(model, "Om"),
        (model_p(model, "n"), model_p(model, "beta")) if model == "dvch" else None)
    e2f, cumf = funcs
    rd = model_p(model, "rd")
    xs, res = [], []
    for tracer, z, obs_list, corr in m.desi_data:
        z = float(z)
        d_m = m.c_speed / model_p(model, "H0") * float(cumf(z))
        d_h = m.c_speed / (model_p(model, "H0") * np.sqrt(float(e2f(z))))
        for kind, val, sigma in obs_list:
            if kind == "DM_rd":
                th = d_m / rd
            elif kind == "DH_rd":
                th = d_h / rd
            else:
                th = (z * d_m**2 * d_h) ** (1.0 / 3.0) / rd
            xs.append(z + (0.02 if model == "dvch" else -0.02))
            res.append((val - th) / sigma)
    ax2.errorbar(xs, res, yerr=1, fmt=marker, ms=5, capsize=3, color=color,
                 label=("DVCH" if model == "dvch" else r"$\Lambda$CDM"))
ax2.axhline(0, color="k", ls="--", lw=0.5)
ax2.set_xlabel("z")
ax2.set_ylabel(r"$\Delta_{\rm BAO} / \sigma$")
ax2.set_title("DESI DR1 BAO Residuals (Tabla 1)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# SNe (residuos con la offset marginalizada analiticamente)
ax3 = axes[2]
for model, color in (("lcdm", "r"), ("dvch", "b")):
    funcs = m.model_functions(
        model, model_p(model, "Om"),
        (model_p(model, "n"), model_p(model, "beta")) if model == "dvch" else None)
    _, cumf = funcs
    integ = cumf(z_hd)
    dl = (1.0 + z_hel) * m.c_speed / model_p(model, "H0") * integ
    mu = 5.0 * np.log10(dl) + 25.0
    delta = mb - mu
    bins = np.linspace(0.01, 1.2, 25)
    idx = np.digitize(z_hd, bins)
    centers, med = [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() >= 5:
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            w = dmb[sel] ** -2
            med.append(float(np.sum(delta[sel] * w) / np.sum(w)))
    offset = float(np.sum(delta / dmb**2) / np.sum(1.0 / dmb**2))
    ax3.plot(centers, np.array(med) - offset, marker, color=color, ms=5,
             label=("DVCH" if model == "dvch" else r"$\Lambda$CDM"))
ax3.axhline(0, color="k", ls="--", lw=0.5)
ax3.set_xlabel("z")
ax3.set_ylabel(r"$\Delta\mu$ (bin) $-$ offset")
ax3.set_title(f"Pantheon$+$ Residuals ({len(z_hd)} SNe)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

fig.suptitle("DVCH Late-Time Residual Diagnostics (CC + DESI BAO + Pantheon$+$)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "dvch_late_time_residuals.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("residuals guardado")
