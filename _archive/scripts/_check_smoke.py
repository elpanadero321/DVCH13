import numpy as np

names = open("/mnt/d/DVCH13-1/dvch_widebeta_smoke.1.txt").readline().lstrip("# ").split()
j_n = names.index("DVCH_n")
j_b = names.index("DVCH_beta")
d = np.loadtxt("/mnt/d/DVCH13-1/dvch_widebeta_smoke.1.txt")
n, b = d[:, j_n], d[:, j_b]
print(f"muestras en smoke: {len(n)}")
print(f"DVCH_n   : min={n.min():.4g} max={n.max():.4g} mediana={np.median(n):.4g}")
print(f"DVCH_beta: min={b.min():.4g} max={b.max():.4g} mediana={np.median(b):.4g}")
print(f"beta explorando la zona nueva (>0.01)? {(b > 0.01).mean():.2f}")
print(f"beta en la zona late-time (~0.4-0.6)? {((b > 0.4) & (b < 0.6)).mean():.2f}")
