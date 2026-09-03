#!/usr/bin/env python3
"""Cross-check generated DVCH numerical outputs."""
import json
import numpy as np
import pandas as pd

print("=== Double-slit: recompute chi^2 from intensities CSV ===")
ds = pd.read_csv("dvch_double_slit_intensities.csv")
required_columns = {"x_exp", "data_obs", "err_obs", "I_standard", "I_DVCH_best"}
missing_columns = required_columns.difference(ds.columns)
if missing_columns:
    raise ValueError(f"Double-slit CSV is missing columns: {sorted(missing_columns)}")
if not np.isfinite(ds[list(required_columns)].to_numpy()).all():
    raise ValueError("Double-slit CSV contains non-finite values")
obs, err = ds["data_obs"].to_numpy(), ds["err_obs"].to_numpy()
ch_qm = float(np.sum(((obs - ds["I_standard"]) / err) ** 2))
ch_dv = float(np.sum(((obs - ds["I_DVCH_best"]) / err) ** 2))
dof = len(ds)
print(f"n_points = {len(ds)}, dof = {dof}")
print(f"chi2_QM   = {ch_qm:.4f}   reduced = {ch_qm/dof:.4f}")
print(f"chi2_DVCH = {ch_dv:.4f}   reduced = {ch_dv/dof:.4f}")
print(f"delta_chi2 (DVCH-QM) = {ch_dv-ch_qm:+.4f}  (sqrt(2*dof)={np.sqrt(2*dof):.2f})")

rep = json.load(open("dvch_double_slit_congruence.json"))
print(f"Stored historical JSON: chi2_QM={rep['fit']['chi2_QM']:.4f}, "
      f"chi2_DVCH={rep['fit']['chi2_DVCH']:.4f}")
print("The current CSV is a deterministic synthetic diagnostic, not Tonomura data.")

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
print(f"max R_hat = {c.R_hat.max():.3f}  (reported convergence table)")
print(f"min ESS   = {c.ESS.min()}  (reported convergence table)")
print(f"implied tau = 24*1200/ESS = {(24*1200/c.ESS).round(2).tolist()}")

e = pd.read_csv("dvch_mcmc_full_evidence.csv")
print("evidence CSV:")
print(e.to_string(index=False))
