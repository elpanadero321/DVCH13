#!/usr/bin/env python3
"""DVCH double-slit consistency test.

 This script generates a reproducible laboratory-scale toy comparison. It does
 not replace a digitized Tonomura data set.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Tonomura-inspired illustrative parameters (not the published electron data)
d = 0.1e-6          # slit separation [m]
lambda_ = 500e-9    # wavelength [m]
I0 = 1.2            # intensity amplitude
Qtilde = -0.0954
alpha = 0.012

# Generate a deterministic synthetic profile with an explicit uncertainty.
rng = np.random.default_rng(20260827)
x_exp = np.linspace(-0.5, 0.5, 100)
x_exp_m = x_exp * 1e-6
arg = np.pi * d * x_exp_m / lambda_
# Avoid 0/0 at the center of the profile.
with np.errstate(divide='ignore', invalid='ignore'):
    sinc = np.sinc(arg / np.pi)
I_standard = I0 * sinc ** 2
err_obs = np.full(len(x_exp), 0.01)
data_obs = np.clip(I_standard + rng.normal(0.0, err_obs), 0.0, None)

# Qtilde is used only as a bounded dimensionless toy amplitude here.
correction = 1.0 + alpha * Qtilde
I_dvch = I_standard * correction

# Calculate, rather than prescribe, the reduced chi-squared values.
dof = len(data_obs)
chi2_standard = float(np.sum(((data_obs - I_standard) / err_obs) ** 2))
chi2_dvch = float(np.sum(((data_obs - I_dvch) / err_obs) ** 2))
chi2_red_standard = chi2_standard / dof
chi2_red_dvch = chi2_dvch / dof

np.savetxt(
    "dvch_double_slit_intensities.csv",
    np.column_stack([x_exp, data_obs, err_obs, I_standard, I_dvch]),
    delimiter=",",
    header="x_exp,data_obs,err_obs,I_standard,I_DVCH_best",
    comments="",
)

# Create figure matching the manuscript request
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].errorbar(x_exp, data_obs, yerr=err_obs, fmt='ro', ms=3,
                 label='Datos sintéticos')
axes[0].plot(x_exp, I_dvch, 'b-', label='DVCH (alpha=0.012)')
axes[0].plot(x_exp, I_standard, color='gray', linestyle='--',
             label='Patrón estándar (alpha=0)')
axes[0].set_title('Coincidencia con DVCH')
axes[0].set_xlabel('Posición x (μm)')
axes[0].set_ylabel('Intensidad I(x) (W/m²)')
axes[0].legend()

resid_plot = (I_dvch - data_obs) / err_obs
axes[1].plot(x_exp, resid_plot, 'g-', label='Residuos DVCH / error')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_ylabel('Residuos / error')
axes[1].set_xlabel('Posición x (μm)')
axes[1].legend()

axes[0].text(0.02, 0.94,
             f'$\\chi^2_{{red}}(DVCH) = {chi2_red_dvch:.3f}$',
             transform=axes[0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'dvch_double_slit.png'), dpi=220)
plt.close(fig)

print('wavelength =', lambda_)
print('slit separation =', d)
print('I0 =', I0)
print('alpha =', alpha)
print('correction =', correction)
print('chi2_red_standard =', chi2_red_standard)
print('chi2_red_dvch =', chi2_red_dvch)
print('output =', os.path.join(FIGDIR, 'dvch_double_slit.png'))
print('csv = dvch_double_slit_intensities.csv')
