"""Construye un covmat afinado a partir de las muestras del burn-in.

La cadena anterior arranco con sigma_beta=0.15 (demasiado grande: tasa de
aceptacion 0.37). Aqui estimamos la covarianza empirica de las muestras de
burn-in, que ya refleja las correlaciones reales, y la usamos como propuesta.
"""
import numpy as np

prefix = "/mnt/d/DVCH13-1/dvch_prod_widebeta"
names = open(f"{prefix}.1.txt").readline().lstrip("# ").split()
print("columnas:", len(names))

j_weight = 0
cols = [names.index(p) for p in names if p not in ("minuslogpost", "minuslogprior",
        "minuslogprior__0", "chi2", "chi2__dvch_plik")]
sampled = [names[c] for c in cols]
print("parametros muestreados:", len(sampled))

data = []
for i in range(1, 5):
    d = np.loadtxt(f"{prefix}.{i}.txt")
    data.append(d[cols])
data = np.vstack(data)
print("muestras totales:", data.shape[0])

# Normalizar pesos
w = data[:, names.index("weight")] if "weight" in names else None

# Estimar covarianza empirica de los parametros muestreados (sin weight)
X = data[:, [i for i, p in enumerate(sampled) if p != "weight"]]
sampled_x = [p for p in sampled if p != "weight"]
C = np.cov(X, rowvar=False)
print("covarianza empirica:", C.shape)

# Suavizado: mezclar con la diagonal para evitar singularidad
C = 0.5 * C + 0.5 * np.diag(np.diag(C))
C = 0.5 * (C + C.T)

ev = np.linalg.eigvalsh(C)
print("autovalor min:", ev.min(), "max:", ev.max())
if ev.min() <= 0:
    C = C + np.eye(len(C)) * (abs(ev.min()) + 1e-10)
    print("  regularizado")

jb = sampled_x.index("DVCH_beta")
jn = sampled_x.index("DVCH_n")
print(f"sigma_beta empirica = {np.sqrt(C[jb, jb]):.5g}")
print(f"sigma_n empirica    = {np.sqrt(C[jn, jn]):.5g}")

# Cabecera con nombres en el formato cobaya
header = "# " + " ".join(sampled_x)
out = "/mnt/d/DVCH13-1/dvch_widebeta_input.covmat"
with open(out, "w") as f:
    f.write(header + "\n")
    np.savetxt(f, C)
print(f"\nEscrito {out} ({len(sampled_x)} parametros)")
print("diagonal:", np.round(np.sqrt(np.diag(C)), 6)[:10])
