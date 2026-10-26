"""Construye dvch_prod_widebeta.covmat con cabecera de nombres (formato cobaya)."""
import numpy as np

names = open("/mnt/d/DVCH13-1/dvch_prod_learned.covmat").readline().strip()
pn = names.lstrip("# ").split()
print("nombres covmat aprendido:", pn)

src = np.loadtxt("/mnt/d/DVCH13-1/dvch_prod_learned.covmat")
assert src.shape == (26, 26), src.shape

j_beta = pn.index("DVCH_beta")
diag = np.sqrt(np.diag(src))
print(f"sigma vieja DVCH_beta = {diag[j_beta]:.6g}")

new = src.copy()
scale = 0.15 / diag[j_beta]
new[j_beta, :] *= scale
new[:, j_beta] *= scale
new = 0.5 * (new + new.T)

ev = np.linalg.eigvalsh(new)
if ev.min() <= 0:
    new = new + np.eye(26) * (abs(ev.min()) + 1e-12)
    print("  regularizado para PD")

print(f"sigma nueva DVCH_beta = {np.sqrt(new[j_beta, j_beta]):.4g}")
print("autovalor minimo:", np.linalg.eigvalsh(new).min())

with open("/mnt/d/DVCH13-1/dvch_widebeta_input.covmat", "w") as f:
    f.write(names + "\n")
    np.savetxt(f, new)
print("Escrito dvch_widebeta_input.covmat con cabecera")
