import numpy as np

# Load the regularized covmat and scale it up so the proposal is wider.
# The initial points were ~25 sigma from the center, so we scale the
# standard deviation by a factor to bring acceptance to ~20-30%.
h = open("dvch_prod_init.covmat").readline().lstrip("#").split()
cov = np.loadtxt("dvch_prod_init.covmat", skiprows=1)
n = cov.shape[0]

# Scale factor: multiply the whole covariance by scale^2 (equiv. scale the
# std by `scale`). 25 sigma -> 5 sigma means scale ~5.
scale = 5.0
cov_scaled = cov * (scale ** 2)

eig = np.linalg.eigvalsh(cov_scaled)
print(f"n_params = {n}")
print(f"scale = {scale}")
print(f"min eig = {eig.min():.3e}, max eig = {eig.max():.3e}")
print(f"condition = {eig.max()/eig.min():.2e}")
print(f"positive definite = {eig.min() > 0}")

with open("dvch_prod_wide.covmat", "w") as out:
    out.write("# " + " ".join(h) + "\n")
    for row in cov_scaled:
        out.write(" ".join(f"{x:.18e}" for x in row) + "\n")
print("Wrote dvch_prod_wide.covmat")