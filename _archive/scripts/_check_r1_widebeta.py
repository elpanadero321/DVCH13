import numpy as np

prefix = "/mnt/d/DVCH13-1/dvch_prod_widebeta"
chains = []
for i in range(1, 5):
    d = np.loadtxt(f"{prefix}.{i}.txt")
    chains.append(d)

names = open(f"{prefix}.1.txt").readline().lstrip("# ").split()
focus = ["H0", "ombh2", "omch2", "tau", "ln10As", "ns", "DVCH_n", "DVCH_beta",
         "chi2", "calib_100T", "calib_217T"]

print("R-1 por parametro (cadenas widebeta, estado actual):")
print(f"{'param':12s} {'R-1':>12s}")
worst = []
for p in focus:
    j = names.index(p)
    means, vars_ = [], []
    for c in chains:
        x = c[:, j]
        w = c[:, 0]  # peso
        m = np.average(x, weights=w)
        v = np.average((x - m) ** 2, weights=w)
        means.append(m)
        vars_.append(v)
    means = np.array(means)
    vars_ = np.array(vars_)
    n = min(len(c) for c in chains)
    W = vars_.mean()
    B = n * means.var(ddof=1)
    r1 = np.sqrt((B / n + W) / W) - 1 if W > 0 else np.nan
    worst.append((p, r1, means.min(), means.max()))
    print(f"{p:12s} {r1:>12.4g}")

print("\nPeores parametros:")
for p, r1, mn, mx in sorted(worst, key=lambda t: -(t[1] if np.isfinite(t[1]) else -1))[:5]:
    print(f"  {p:12s} R-1={r1:10.4g}  rango de medias: [{mn:.4g}, {mx:.4g}]")

# Distribucion de beta por cadena
jb = names.index("DVCH_beta")
print("\nDVCH_beta por cadena (media, min, max):")
for i, c in enumerate(chains, 1):
    x = c[:, jb]
    print(f"  cadena {i}: media={x.mean():.4g} min={x.min():.4g} max={x.max():.4g}")
