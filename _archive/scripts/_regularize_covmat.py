import numpy as np

h = open("dvch_calib_empirical.covmat").readline().lstrip("#").split()
cov = np.loadtxt("dvch_calib_empirical.covmat", skiprows=1)
n = cov.shape[0]

# Ridge regularization: add a fraction of the diagonal to the diagonal
# to improve conditioning. Standard approach: cov_reg = cov + eps * diag(diag(cov))
diag = np.diag(cov)
# Use a small ridge: 1e-3 of the median diagonal as floor
floor = 1e-3 * np.median(diag)
cov_reg = cov.copy()
np.fill_diagonal(cov_reg, diag + floor)

eig = np.linalg.eigvalsh(cov_reg)
print("min eigenvalue (reg):", eig.min())
print("condition number (reg):", eig.max() / eig.min())
print("positive definite:", eig.min() > 0)

# Also check: is the floor too small? Try a few floors
for f in [1e-4, 1e-3, 1e-2, 1e-1]:
    c = cov.copy()
    np.fill_diagonal(c, diag + f * np.median(diag))
    e = np.linalg.eigvalsh(c)
    print(f"floor={f}: cond={e.max()/e.min():.2e}, min_eig={e.min():.2e}")

# Write the regularized covariance (floor=1e-3).
# NOTE: use a name that does NOT collide with the Cobaya output prefix
# (Cobaya's force=True deletes '<prefix>.covmat' before loading it).
with open("dvch_prod_init.covmat", "w") as out:
    out.write("# " + " ".join(h) + "\n")
    for row in cov_reg:
        out.write(" ".join(f"{x:.18e}" for x in row) + "\n")
print("\nWrote dvch_prod_init.covmat (regularized)")