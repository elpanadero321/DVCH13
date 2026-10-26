"""Covmat afinado: mezcla el empirico del burn-in con el aprendido estable.

Con solo ~108 muestras el empirico puro es ruidoso; mezclarlo con el
covmat aprendido de la cadena anterior (26x26, mismo orden de params)
mantiene la estabilidad y mejora la escala de DVCH_beta.
"""
import numpy as np

header = open("/mnt/d/DVCH13-1/dvch_widebeta_input.covmat").readline().strip()
pn = header.lstrip("# ").split()
assert len(pn) == 26, len(pn)

learned = np.loadtxt("/mnt/d/DVCH13-1/dvch_prod_learned.covmat")
emp = np.loadtxt("/mnt/d/DVCH13-1/dvch_widebeta_input.covmat")
assert learned.shape == emp.shape == (26, 26)

jb = pn.index("DVCH_beta")
jn = pn.index("DVCH_n")
print(f"learned: sigma_beta={np.sqrt(learned[jb,jb]):.5g} sigma_n={np.sqrt(learned[jn,jn]):.5g}")
print(f"empirico: sigma_beta={np.sqrt(emp[jb,jb]):.5g} sigma_n={np.sqrt(emp[jn,jn]):.5g}")

# Mezcla: 60% empirico (aporta las correlaciones y escala de beta nueva)
# + 40% aprendido (aporta estabilidad del resto de parametros)
C = 0.6 * emp + 0.4 * learned
C = 0.5 * (C + C.T)

ev = np.linalg.eigvalsh(C)
print(f"mezcla: sigma_beta={np.sqrt(C[jb,jb]):.5g} sigma_n={np.sqrt(C[jn,jn]):.5g}")
print(f"autovalor min={ev.min():.3e} max={ev.max():.3e}")
if ev.min() <= 0:
    C = C + np.eye(26) * (abs(ev.min()) + 1e-10)
    print("  regularizado para PD")

out = "/mnt/d/DVCH13-1/dvch_widebeta_input.covmat"
with open(out, "w") as f:
    f.write(header + "\n")
    np.savetxt(f, C)
print(f"\nEscrito {out}")
