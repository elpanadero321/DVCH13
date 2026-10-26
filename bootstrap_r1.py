#!/usr/bin/env python3
# Test crucial: el R-1 multivariado de cobaya con m cadenas en d dimensiones
# tiene un SUELO DE RUIDO. Si m es chico vs d, la cov de medias es de rango m-1
# y sus autovalores se dispersan aunque las cadenas esten perfectamente convergidas.
#
# Metodologia: reagrupar las muestras existentes en m cadenas artificiales
# asignandolas aleatoriamente (=> convergencia "perfecta" por construccion),
# y calcular el R-1 que el estimador devolveria. Eso es el suelo de ruido.
import glob, numpy as np

files = sorted(glob.glob('/mnt/d/DVCH13-1/dvch_prod_widebeta.[0-9]*.txt'))
chains, names = [], None
for f in files:
    with open(f) as fh:
        header, rows = None, []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                header = line.lstrip('#').split()
                continue
            rows.append([float(x) for x in line.split()])
    names = header
    chains.append(np.array(rows))

try:
    i_end = names.index('minuslogprior')
except ValueError:
    i_end = len(names)
param_idx = list(range(2, i_end))
n = min(len(c) for c in chains)
m = len(chains)

def cobaya_rminus1(means, covs):
    mean_of_covs = np.average(covs, axis=0)
    cov_of_means = np.atleast_2d(np.cov(means.T))
    d = np.sqrt(np.diag(cov_of_means))
    corr_of_means = (cov_of_means / d).T / d
    norm_mean_of_covs = (mean_of_covs / d).T / d
    L = np.linalg.cholesky(norm_mean_of_covs)
    Linv = np.linalg.inv(L)
    M = Linv.dot(corr_of_means).dot(Linv.T)
    return max(np.abs(np.linalg.eigvalsh(M)))

# --- 1) R-1 real (replica cobaya) ---
use_first = n // 2
means = np.zeros((m, len(param_idx)))
covs = np.zeros((m, len(param_idx), len(param_idx)))
for k, c in enumerate(chains):
    v = c[len(c) - n:, param_idx][use_first:]
    means[k] = v.mean(axis=0)
    covs[k] = np.cov(v.T)
r_real = cobaya_rminus1(means, covs)
print(f"R-1 real (4 cadenas)          = {r_real:.4f}")

# --- 2) suelo de ruido: reagrupar muestras aleatoriamente en m cadenas ---
rng = np.random.default_rng(0)
pooled = np.vstack([c[len(c) - n:, param_idx] for c in chains])
for m_test in [4, 6, 8, 12, 16]:
    vals = []
    for trial in range(6):
        perm = rng.permutation(len(pooled))
        fake = pooled[perm]
        per = len(fake) // m_test
        fm = np.array([fake[i*per:(i+1)*per][use_first:].mean(axis=0) for i in range(m_test)])
        fc = np.array([np.cov(fake[i*per:(i+1)*per][use_first:].T) for i in range(m_test)])
        vals.append(cobaya_rminus1(fm, fc))
    print(f"  suelo de ruido con {m_test:2d} cadenas = {np.mean(vals):.4f} (+-{np.std(vals):.4f})")

print("\nInterpretacion:")
print("  Si R-1 real ~ suelo de ruido => las cadenas YA estan convergidas y")
print("  el 0.19 es un artefacto del estimador con pocas cadenas.")
print("  Si R-1 real >> suelo de ruido => la no-convergencia es real.")
