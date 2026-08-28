#!/usr/bin/env python3
"""Cross-check DVCH.tex claims against the analysis outputs (read-only)."""
import json
import numpy as np
import pandas as pd

print("=== Double-slit: recompute chi^2 from intensities CSV ===")
ds = pd.read_csv("dvch_double_slit_intensities.csv")
obs, err = ds["data_obs"].to_numpy(), ds["err_obs"].to_numpy()
ch_qm = float(np.sum(((obs - ds["I_standard"]) / err) ** 2))
ch_dv = float(np.sum(((obs - ds["I_DVCH_best"]) / err) ** 2))
dof = len(ds) - 1
print(f"n_points = {len(ds)}, dof = {dof}")
print(f"chi2_QM   = {ch_qm:.4f}   reduced = {ch_qm/dof:.4f}")
print(f"chi2_DVCH = {ch_dv:.4f}   reduced = {ch_dv/dof:.4f}")
print(f"delta_chi2 (DVCH-QM) = {ch_dv-ch_qm:+.4f}  (sqrt(2*dof)={np.sqrt(2*dof):.2f})")

rep = json.load(open("dvch_double_slit_congruence.json"))
print(f"JSON says: chi2_QM={rep['fit']['chi2_QM']:.4f}, chi2_DVCH={rep['fit']['chi2_DVCH']:.4f}, "
      f"red_QM={rep['fit']['chi2_red_QM']:.4f}, red_DVCH={rep['fit']['chi2_red_DVCH']:.4f}, "
      f"dchi2={rep['fit']['delta_chi2']:+.4f}")
print(f"NOTE: script's 'chi2' = 0.5*sum(resid^2) = -ln L, i.e. HALF the standard chi2")
print(f"beta_eff best = {rep['fit']['beta_eff_best_m2']:.4e} m^2, S = {rep['fit']['suppression_amplitude_S']:.6f}")

print()
print("=== Full MCMC: chains, summary, convergence, evidence ===")
ch = pd.read_csv("dvch_mcmc_chains_full.csv")
print(f"chains_full rows = {len(ch)} (doc claims 1056), cols = {list(ch.columns)}")
for p in ["Om", "n", "beta", "H0"]:
    v = ch[p].to_numpy()
    print(f"  {p:5s} median={np.median(v):.4f}  68%=[{np.percentile(v,16):.4f},{np.percentile(v,84):.4f}]  "
          f"95%=[{np.percentile(v,2.5):.4f},{np.percentile(v,97.5):.4f}]")

s = pd.read_csv("dvch_mcmc_full_summary.csv")
print("summary CSV:")
print(s[["parameter", "median", "q16", "q84", "q025", "q975"]].to_string(index=False))

c = pd.read_csv("dvch_mcmc_full_convergence.csv")
print("convergence CSV:")
print(c.to_string(index=False))
print(f"max R_hat = {c.R_hat.max():.3f}  (doc claims 1.059)")
print(f"min ESS   = {c.ESS.min()}  (doc claims 379)")
print(f"implied tau = 24*1200/ESS = {(24*1200/c.ESS).round(2).tolist()}")

e = pd.read_csv("dvch_mcmc_full_evidence.csv")
print("evidence CSV:")
print(e.to_string(index=False))
