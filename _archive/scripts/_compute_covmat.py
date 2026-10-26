import numpy as np
import pandas as pd

h = open("dvch_calib.1.txt").readline().lstrip("#").split()
f = pd.read_csv("dvch_calib.1.txt", sep=r"\s+", names=h, skiprows=1)

# Parameter columns (exclude weight, minuslogpost, minuslogprior*, chi2*)
params = [c for c in h
          if c not in {"weight", "minuslogpost"}
          and not c.startswith("minuslogprior")
          and not c.startswith("chi2")]
print("n_params:", len(params))

# Expand by weight to get the actual accepted samples
vals = f[params].to_numpy(dtype=float)
weights = f["weight"].to_numpy(dtype=int)
expanded = np.repeat(vals, weights, axis=0)
print("expanded samples:", expanded.shape[0])

# Empirical covariance
cov = np.cov(expanded, rowvar=False)
print("cov shape:", cov.shape)
print("cov diagonal (variances):")
for i, p in enumerate(params):
    print(f"  {p}: {cov[i,i]:.6e}")

# Check off-diagonal (correlations)
corr = np.corrcoef(expanded, rowvar=False)
print("\nTop 5 correlations (|r| > 0.3):")
pairs = []
for i in range(len(params)):
    for j in range(i+1, len(params)):
        pairs.append((abs(corr[i,j]), params[i], params[j], corr[i,j]))
pairs.sort(reverse=True)
for r, a, b, c in pairs[:5]:
    print(f"  {a} <-> {b}: r={c:.3f}")

# Save covariance in Cobaya format (header + matrix)
with open("dvch_calib_empirical.covmat", "w") as out:
    out.write("# " + " ".join(params) + "\n")
    for row in cov:
        out.write(" ".join(f"{x:.18e}" for x in row) + "\n")
print("\nWrote dvch_calib_empirical.covmat")