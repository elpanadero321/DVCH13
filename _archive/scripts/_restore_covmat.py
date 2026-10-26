"""Reconstruye el covmat con la receta que funciono en la primera corrida.

Corrida 1 (covmat aprendido + sigma_beta=0.15): aceptacion 0.37, cadenas
sanas, explorando beta en [0.03, 0.99]. El unico problema fue que
learn_proposal_Rminus1_max=2.0 impedia aprender el covmat con R-1=375.
Eso ya se soluciono subiendo el umbral a 30.

Corrida 2 (covmat empirico de 108 muestras de burn-in): 0 aceptaciones,
correlaciones ruidosas. Descartada.

Aqui restauramos la receta de la corrida 1.
"""
import numpy as np

header = open("/mnt/d/DVCH13-1/dvch_prod_learned.covmat").readline().strip()
pn = header.lstrip("# ").split()
assert len(pn) == 26

src = np.loadtxt("/mnt/d/DVCH13-1/dvch_prod_learned.covmat")
jb = pn.index("DVCH_beta")
diag = np.sqrt(np.diag(src))

new = src.copy()
scale = 0.15 / diag[jb]
new[jb, :] *= scale
new[:, j_beta_col := jb] *= scale  # noqa: F841
new = 0.5 * (new + new.T)

ev = np.linalg.eigvalsh(new)
if ev.min() <= 0:
    new = new + np.eye(26) * (abs(ev.min()) + 1e-12)
    print("regularizado")

print(f"sigma_beta = {np.sqrt(new[jb, jb]):.5g}")
print(f"autovalor min = {np.linalg.eigvalsh(new).min():.3e}")

out = "/mnt/d/DVCH13-1/dvch_widebeta_input.covmat"
with open(out, "w") as f:
    f.write(header + "\n")
    np.savetxt(f, new)
print(f"Escrito {out} (receta corrida 1)")
