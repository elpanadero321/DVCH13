#!/usr/bin/env python3
# Extrae el minimo chi2 (best fit) de la cadena de DVCH ya convergida.
import glob, numpy as np

files = sorted(glob.glob('/mnt/d/DVCH13-1/dvch_prod_widebeta5.[0-9]*.txt'))
rows_all, names = [], None
for f in files:
    with open(f) as fh:
        header = None
        rows = []
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith('#'):
                header = line.lstrip('#').split(); continue
            rows.append([float(x) for x in line.split()])
    names = header
    rows_all.extend(rows)

data = np.array(rows_all)
# columnas: weight minuslogpost ... minuslogprior minuslogprior__0 chi2 chi2__dvch_plik
i = {n: k for k, n in enumerate(names)}
chi2_total = data[:, i['chi2']]
chi2_plik  = data[:, i['chi2__dvch_plik']]
mlp = data[:, i['minuslogpost']]

k_total = int(np.argmin(mlp))          # min logpost = max posterior
k_chi2  = int(np.argmin(chi2_total))   # min chi2 puro

print(f"muestras totales: {len(data)}")
print()
print("Best-fit por maxima posterior (MAP):")
print(f"  chi2_total = {chi2_total[k_total]:.2f}")
print(f"  chi2_plik  = {chi2_plik[k_total]:.2f}")
print(f"  H0={data[k_total, i['H0']]:.3f}  DVCH_n={data[k_total, i['DVCH_n']]:.4f}  DVCH_beta={data[k_total, i['DVCH_beta']]:.4f}")
print()
print("Best-fit por minimo chi2 (maxima verosimilitud):")
print(f"  chi2_total = {chi2_total[k_chi2]:.2f}")
print(f"  chi2_plik  = {chi2_plik[k_chi2]:.2f}")
print(f"  H0={data[k_chi2, i['H0']]:.3f}  DVCH_n={data[k_chi2, i['DVCH_n']]:.4f}  DVCH_beta={data[k_chi2, i['DVCH_beta']]:.4f}")
print()
print(f"percentil chi2_total: min={chi2_total.min():.2f} 5%={np.percentile(chi2_total,5):.2f} mediana={np.median(chi2_total):.2f}")
