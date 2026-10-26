#!/usr/bin/env python3
# Replica exacta del Rminus1 multivariado de cobaya y diagnostica la direccion
# (combinacion de parametros) que domina la no-convergencia.
import glob, numpy as np

files = sorted(glob.glob('/mnt/d/DVCH13-1/dvch_prod_widebeta5.[0-9]*.txt'))
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
pnames = [names[i] for i in param_idx]

# alinear por la cola
n = min(len(c) for c in chains)
m = len(chains)

# cobaya: usa la segunda mitad de cada cadena (use_first = n/2)
use_first = n // 2
means = np.zeros((m, len(param_idx)))
covs = np.zeros((m, len(param_idx), len(param_idx)))
for k, c in enumerate(chains):
    v = c[len(c) - n:, param_idx]
    v = v[use_first:]
    means[k] = v.mean(axis=0)
    covs[k] = np.cov(v.T)

mean_of_covs = np.average(covs, axis=0)  # cobaya no pondera por N aqui
cov_of_means = np.atleast_2d(np.cov(means.T))
d = np.sqrt(np.diag(cov_of_means))
corr_of_means = (cov_of_means / d).T / d
norm_mean_of_covs = (mean_of_covs / d).T / d

L = np.linalg.cholesky(norm_mean_of_covs)
Linv = np.linalg.inv(L)
M = Linv.dot(corr_of_means).dot(Linv.T)
eigvals, eigvecs = np.linalg.eigh(M)
i_max = int(np.argmax(np.abs(eigvals)))
Rminus1 = eigvals[i_max]
print(f"Rminus1 multivariado (replica cobaya) = {Rminus1:.4f}")
print(f"cobaya reporta                        = 0.1922\n")

# direccion dominante: que parametros contribuyen a ese autovector
v = np.abs(eigvecs[:, i_max])
v = v / v.max()
order = np.argsort(v)[::-1]
print("Direccion dominante de la no-convergencia (params con mas peso):")
for i in order[:12]:
    print(f"  {pnames[i]:22s} {v[i]:.3f}")

print("\nAutovalores (todos):")
print("  " + ", ".join(f"{x:.4f}" for x in sorted(eigvals, reverse=True)))
