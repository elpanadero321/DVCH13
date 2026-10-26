"""Veredicto sobre DVCH_beta con las muestras actuales de widebeta.

Contexto: el perfil de chi2 de Planck es casi plano en beta (dchi2~0.16 en
[1e-4, 0.9]), asi que el posterior deberia ser casi uniforme sobre el prior
[0,1]. Este script lo confirma cuantitativamente y da el veredicto sobre la
teoria DVCH con el prior corregido.
"""
import numpy as np
from getdist import loadMCSamples

s = loadMCSamples("/mnt/d/DVCH13-1/dvch_prod_widebeta",
                  settings={"ignore_rows": 0})
names = [p.name for p in s.getParamNames().names]
idx = {n: i for i, n in enumerate(names)}
d = s.samples

print("=" * 74)
print("VEREDICTO DVCH CON PRIOR DE beta CORREGIDO (widebeta)")
print("=" * 74)
print(f"Muestras totales: {s.numrows}")

print("\n--- Comparacion posterior vs prior (poder informativo del CMB) ---")
for pname, lo, hi, lbl in [("DVCH_n", 0.01, 0.25, "n"),
                           ("DVCH_beta", 0.0, 1.0, "beta")]:
    if pname not in idx:
        print(f"  {lbl}: parametro no encontrado")
        continue
    x = d[:, idx[pname]]
    prior_sigma = (hi - lo) / np.sqrt(12.0)
    post_sigma = s.std(pname)
    ratio = post_sigma / prior_sigma
    q025, q50, q975 = np.percentile(x, [2.5, 50, 97.5])
    print(f"\n  {lbl}:")
    print(f"    media   = {s.mean(pname):.5g}")
    print(f"    mediana = {q50:.5g}")
    print(f"    IC 95%  = [{q025:.5g}, {q975:.5g}]")
    print(f"    sigma prior = {prior_sigma:.5g}, sigma post = {post_sigma:.5g}")
    print(f"    ratio post/prior = {ratio:.3f}", end=" ")
    if ratio > 0.9:
        print("-> NO INFORMA (posterior ~= prior)")
    elif ratio > 0.6:
        print("-> informacion parcial")
    else:
        print("-> CMB acota significativamente")

# Distancia al limite LCDM
print("\n--- Distancia al limite LCDM (n->0.01, beta->0) ---")
for pname, lcdm, lbl in [("DVCH_n", 0.01, "n"), ("DVCH_beta", 0.0, "beta")]:
    if pname in idx:
        m, sd = s.mean(pname), s.std(pname)
        print(f"  {lbl}: {m:.5g} +/- {sd:.5g}  ->  {(m-lcdm)/sd:.2f} sigma del limite LCDM")

# R-1 por parametro (diagnostico de mezcla)
print("\n--- R-1 por parametro (Gelman-Rubin manual) ---")
offs = list(s.chain_offsets)
r1s = {}
for pname in ["H0", "ombh2", "omch2", "tau", "ln10As", "ns", "DVCH_n", "DVCH_beta"]:
    if pname not in idx:
        continue
    j = idx[pname]
    means, vars_ = [], []
    for k in range(len(offs) - 1):
        x = d[offs[k]:offs[k + 1], j]
        w = s.weights[offs[k]:offs[k + 1]]
        m = np.average(x, weights=w)
        means.append(m)
        vars_.append(np.average((x - m) ** 2, weights=w))
    means, vars_ = np.array(means), np.array(vars_)
    n = min(offs[k + 1] - offs[k] for k in range(len(offs) - 1))
    W = vars_.mean()
    B = n * means.var(ddof=1)
    r1s[pname] = np.sqrt((B / n + W) / W) - 1 if W > 0 else np.nan
    print(f"  {pname:12s} R-1 = {r1s[pname]:8.4f}")

print("\n" + "=" * 74)
