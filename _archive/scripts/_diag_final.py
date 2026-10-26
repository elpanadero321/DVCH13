import numpy as np
from getdist import loadMCSamples

PREFIX = "/mnt/d/DVCH13-1/dvch_prod"

s = loadMCSamples(PREFIX, settings={"ignore_rows": 0})
offs = list(s.chain_offsets)
print("chain offsets:", offs)

par_names = [p.name for p in s.getParamNames().names]
idx = {n: i for i, n in enumerate(par_names)}

data = s.samples          # (N, npar) sin pesos
weights = s.weights

print("=" * 72)
print("DIAGNOSTICO DE CADENAS - DVCH_PROD (high-l nuisance smoke)")
print("=" * 72)
print(f"Muestras totales: {s.numrows}  |  Parametros: {len(par_names)}")

# --- R-1 por parametro ---
print("\n--- R-1 (Gelman-Rubin) por parametro ---")
r1_dict = {}
for name in par_names:
    j = idx[name]
    means, vars_ = [], []
    for k in range(len(offs) - 1):
        x = data[offs[k]:offs[k + 1], j]
        w = weights[offs[k]:offs[k + 1]]
        m = np.average(x, weights=w)
        v = np.average((x - m) ** 2, weights=w)
        means.append(m)
        vars_.append(v)
    means = np.array(means)
    vars_ = np.array(vars_)
    n = min(offs[k + 1] - offs[k] for k in range(len(offs) - 1))
    W = vars_.mean()
    B = n * means.var(ddof=1)
    r1 = (np.sqrt((B / n + W) / W) - 1.0) if W > 0 else np.nan
    r1_dict[name] = r1

for name, r1 in r1_dict.items():
    flag = "OK" if r1 < 0.03 else ("aceptable" if r1 < 0.05 else
          ("regular" if r1 < 0.10 else "REVISAR"))
    print(f"  {name:22s} R-1 = {r1:8.4f}  [{flag}]")

r1_arr = np.array([v for v in r1_dict.values() if np.isfinite(v)])
print(f"\n  R-1 maximo  = {r1_arr.max():.4f}  ({max(r1_dict, key=lambda k: r1_dict[k] if np.isfinite(r1_dict[k]) else -1)})")
print(f"  R-1 medio   = {r1_arr.mean():.4f}")
print(f"  R-1 global getdist (ortogonalizado) = {s.getGelmanRubin():.4f}")

# --- Bondad del ajuste ---
print("\n--- Bondad del ajuste ---")
try:
    best = s.getBestFit()
    print(f"  chi2_min         = {best.chi2:.3f}")
    print(f"  minuslogpost_min = {best.logSample:.4f}")
except Exception as e:
    print(f"  (best fit: {e})")

# chi2 por cadena (columna chi2 si existe)
if "chi2" in idx:
    j = idx["chi2"]
    per = [data[offs[k]:offs[k+1], j].min() for k in range(len(offs) - 1)]
    print(f"  chi2_min por cadena = {['%.3f' % x for x in per]}")

# --- KL divergence aprox ---
print("\n--- Divergencia KL aprox. entre cadenas ---")
means = np.array([np.average(data[offs[k]:offs[k+1], :], axis=0,
                             weights=weights[offs[k]:offs[k+1]])
                  for k in range(len(offs) - 1)])
grand = means.mean(axis=0)
var_mean = s.getVars()
with np.errstate(divide="ignore", invalid="ignore"):
    kl = 0.5 * np.nansum((means - grand) ** 2 / var_mean)
print(f"  KL approx = {kl:.5f}")

# --- Restricciones ---
focus = ["H0", "ombh2", "omch2", "tau", "ln10As", "ns", "DVCH_n", "DVCH_beta"]
print("\n--- Restricciones (media +/- 1 sigma) ---")
for name in focus:
    if name in idx:
        m, sd = s.mean(name), s.std(name)
        print(f"  {name:10s} = {m:12.6g} +/- {sd:10.6g}")
    else:
        print(f"  {name:10s} (no existe)")

print("\n" + "=" * 72)
