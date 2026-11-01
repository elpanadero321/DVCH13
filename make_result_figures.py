#!/usr/bin/env python3
# Genera las dos figuras que faltan en el paper:
#  (1) corner plot: posteriors late-time LCDM vs DVCH (datos reales)
#  (2) comparacion de modelos: log Z (LCDM, w0waCDM, DVCH) con barreas de error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/mnt/d/DVCH13-1'

# ---------------- (1) corner plot LCDM vs DVCH ----------------
lcdm = np.load(f'{BASE}/dvch_real_chain_lcdm.npy')   # (N,3): Om, H0, M
dvch = np.load(f'{BASE}/dvch_real_chain_dvch.npy')   # (N,5): Om, H0, M, n, beta

params = [
    (r'$\Omega_m$',   (0.22, 0.34)),
    (r'$H_0$',        (66.5, 70.5)),
    (r'$M$',          (-19.7, -19.1)),
    (r'$n$',          (-0.1, 1.0)),
    (r'$\beta$',      (-0.2, 2.2)),
]

fig, axes = plt.subplots(5, 5, figsize=(11, 10))
for i in range(5):
    for j in range(5):
        ax = axes[i, j]
        if j > i:
            ax.axis('off')
            continue
        if i >= 3 and j < 3:
            # solo DVCH tiene (n, beta): omitir los paneles LCDM-only
            ax.axis('off')
            continue
        if i < 3:
            x = lcdm[:, j] if j < 3 else None
            y = lcdm[:, i] if i < 3 else None
        else:
            x = dvch[:, j]; y = dvch[:, i]
        if i == j:
            # histograma diagonal
            if i < 3:
                ax.hist(lcdm[:, i], bins=60, density=True, alpha=0.5, color='C0')
            ax.hist(dvch[:, i], bins=60, density=True, alpha=0.5, color='C1')
            ax.set_xlim(*params[i][1])
        else:
            ax.plot(x, y, ',', alpha=0.02, color='C1')
            ax.set_xlim(*params[j][1])
            ax.set_ylim(*params[i][1])
        if j > 0:
            ax.set_yticks([])
        if i < 4:
            ax.set_xticks([])
    if i == 4:
        for j in range(5):
            axes[i, j].set_xlabel(params[j][0], fontsize=11)
    axes[i, 0].set_ylabel(params[i][0], fontsize=11)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='C0', label=r'$\Lambda$CDM'),
                    Patch(color='C1', label='DVCH')],
           loc='upper right', fontsize=12)
fig.suptitle('Late-time posteriors (Pantheon$+$ 1701 SNe + DESI DR1 BAO + CC, real data)',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{BASE}/figures/dvch_latetime_corner.png', dpi=150, bbox_inches='tight')
print('corner plot guardado')

# ---------------- (2) comparacion de modelos ----------------
names = [r'$\Lambda$CDM', r'$w_0w_a$CDM', r'DVCH']
logz = [-928.09, -920.31, -917.57]
errs = [0.27, 0.31, 0.29]

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ['C0', 'C2', 'C3']
bars = ax.bar(names, logz, yerr=errs, color=colors, capsize=6,
              edgecolor='k', linewidth=0.6)
for b, v in zip(bars, logz):
    ax.text(b.get_x() + b.get_width()/2, v + 0.25, f'{v:.2f}',
            ha='center', fontsize=11)
ax.axhline(logz[2], ls='--', lw=1, color='C3', alpha=0.5)
ax.annotate(r'$\Delta\ln Z$(DVCH$-\Lambda$CDM) $=+10.53$',
            xy=(1, -917.9), fontsize=11, color='C3')
ax.set_ylabel(r'$\ln Z$ (nested sampling)', fontsize=12)
ax.set_title('Bayesian evidence on Pantheon$+$ + DESI DR1 BAO + CC (real data)',
             fontsize=13)
ax.set_ylim(min(logz) - 8, max(logz) + 4)
fig.tight_layout()
fig.savefig(f'{BASE}/figures/dvch_model_comparison.png', dpi=150, bbox_inches='tight')
print('model comparison guardado')
