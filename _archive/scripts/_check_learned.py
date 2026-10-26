import numpy as np

path = "/mnt/d/DVCH13-1/dvch_prod_widebeta.covmat"
header = open(path).readline().lstrip("# ").split()
C = np.loadtxt(path)
print(f"covmat aprendido: {C.shape} x {len(header)}")
jb = header.index("DVCH_beta")
jn = header.index("DVCH_n")
print(f"sigma_DVCH_beta aprendida = {np.sqrt(C[jb, jb]):.5g}")
print(f"sigma_DVCH_n aprendida    = {np.sqrt(C[jn, jn]):.5g}")
print("autovalor min:", np.linalg.eigvalsh(C).min())
