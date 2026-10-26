import os
import numpy as np

for root in ["dvch_prod_wide", "dvch_prod_learned", "dvch_prod_init"]:
    path = f"/mnt/d/DVCH13-1/{root}.covmat"
    if not os.path.exists(path):
        print(root, "-> NO EXISTE")
        continue
    c = np.loadtxt(path)
    pn_path = path + ".paramnames"
    print(f"\n=== {root}  shape={c.shape} ===")
    if os.path.exists(pn_path):
        pn = [l.split()[0] for l in open(pn_path) if l.strip()]
        print("params:", pn)
        for target in ("DVCH_beta", "DVCH_n"):
            if target in pn:
                j = pn.index(target)
                print(f"  {target}: sigma = {np.sqrt(c[j, j]):.6g}")
    else:
        print("  (sin .paramnames) diag:", np.sqrt(np.diag(c)))
