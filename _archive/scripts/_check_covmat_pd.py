import numpy as np

# Load the empirical covmat we wrote
h = open("dvch_calib_empirical.covmat").readline().lstrip("#").split()
cov = np.loadtxt("dvch_calib_empirical.covmat", skiprows=1)
print("shape:", cov.shape, "n_params:", len(h))

# Check positive-definiteness
eig = np.linalg.eigvalsh(cov)
print("min eigenvalue:", eig.min())
print("max eigenvalue:", eig.max())
print("positive definite:", eig.min() > 0)

# Condition number
print("condition number:", eig.max() / eig.min())

# Check symmetry
print("symmetric:", np.allclose(cov, cov.T))