"""Scratch verification of the DVCH dynamical-system claims (temporary)."""
import time
import numpy as np

n = 0.3


def F(x, y, u, n=n):
    d = x + n * y
    if d == 0:
        return np.array([0.0, 0.0, -4.0 * u])
    return np.array([-3 * x * x / d, -3 * n * x * y / d, -4 * u])


print("== F at the claimed 'matter point' (x,0,0), x>0 ==")
for x in (1.0, 0.3, 0.05):
    print(f"  x={x:5.2f} y=0 u=0 -> F={F(x, 0, 0)}   (non-zero => NOT an equilibrium)")

print("\n== F at the claimed 'radiation point' (0,0,u) : 0/0 in f1,f2 ==")
d = 0.0
print("  x+ny =", d, "-> vector field undefined (removable limit depends on path)")

print("\n== path dependence of the limiting Jacobian at the origin ==")


def J(x, y, n=n):
    d = x + n * y
    return np.array([[-3 * x * (x + 2 * n * y) / d**2, 3 * n * x * x / d**2, 0],
                     [-3 * n * n * y * y / d**2, -3 * n * x * x / d**2, 0],
                     [0, 0, -4]])


eps = 1e-9
paths = {
    "along y=0 (matter ray)": lambda e: (e, 0.0),
    "along y=kx (k=1)": lambda e: (e, e),
    "along leaf y=C x^n (C=2)": lambda e: (e, 2 * e**n),
}
for label, f in paths.items():
    x, y = f(eps)
    ev = np.linalg.eigvals(J(x, y))
    print(f"  {label:28s} J|eps -> \n{np.round(J(x, y), 4)}\n    eigs={np.round(ev,4)}")

print("\n== whole y-axis is an equilibrium set of the extended (unphysical) field ==")
for y in (0.2, 1.0, 3.0):
    print(f"  F(0,{y},0) =", F(0.0, y, 0.0))

print("\n== analytic leaf solution: x' = -3x^2/(x+n C x^n) ==")
C = 2.0
x0 = 0.3
N = np.linspace(0, 200, 400001)
h = N[1] - N[0]
x = x0
u = 0.9e-4
ys = []
xs = []
for i in range(1, len(N)):
    k1 = -3 * x * x / (x + n * C * x**n)
    k2 = -3 * (x + h / 2 * k1) ** 2 / ((x + h / 2 * k1) + n * C * (x + h / 2 * k1) ** n)
    k3 = -3 * (x + h / 2 * k2) ** 2 / ((x + h / 2 * k2) + n * C * (x + h / 2 * k2) ** n)
    k4 = -3 * (x + h * k3) ** 2 / ((x + h * k3) + n * C * (x + h * k3) ** n)
    x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    xs.append(x)
xs = np.array([x0] + list(xs))
# exact: d x^{n-1}/dN = 3(1-n)C/(n) ... verify
lhs = xs ** (n - 1)
slope_fit = np.polyfit(N, lhs, 1)
pred = 3 * (1 - n) / (n * C)
print(f"  x^(n-1) linear in N? fit slope={slope_fit[0]:.8f} predicted={pred:.8f}"
      f"  rel.err={abs(slope_fit[0]-pred)/pred:.2e}")
asym = (x0 ** (1 - n) + pred * N) ** (-1.0 / (1 - n))
print(f"  max rel err of x(N) vs power-law formula over N in [50,200]: "
      f"{np.max(np.abs(xs[200/1:] / asym[200/1:] - 1)) if False else np.max(np.abs(xs[100000:]/asym[100000:]-1)):.3e}")
print(f"  x(N=200)={xs[-1]:.3e}  y=Cx^n={C*xs[-1]**n:.3e}  u=u0 e^-4N={u*np.exp(-4*N[-1]):.3e}")

print("\n== closed-form root of the implicit Friedmann equation vs fixed point iteration ==")
Omr = 9e-5


def e2_fp(z, Om, nn, beta, iters=30, tol=1e-12):
    OL = 1 - Om - Omr
    opz = 1 + z
    rad = Omr * opz**4
    Omz = Om * opz**3
    E2 = rad + Omz + OL
    for _ in range(iters):
        OLz = OL * (Omz / Om)**nn * (1 + beta) / (1 + beta * E2)
        new = rad + Omz + OLz
        if abs(new - E2) < tol:
            break
        E2 = new
    return E2


def e2_closed(z, Om, nn, beta):
    OL = 1 - Om - Omr
    opz = 1 + z
    A = Omr * opz**4 + Om * opz**3
    B = OL * (Om * opz**3 / Om)**nn * (1 + beta)
    if beta < 1e-14:
        return A + B
    disc = (1 - beta * A) ** 2 + 4 * beta * B
    return 2 * B / (1 - beta * A + np.sqrt(disc))


rng = np.random.default_rng(0)
worst = 0.0
for _ in range(400):
    Om = rng.uniform(0.05, 0.6)
    nn = rng.uniform(0.01, 0.6)
    beta = 10 ** rng.uniform(-8, -0.5)
    for z in (0.0, 0.1, 0.5, 1.0, 2.5, 5.0):
        a, b = e2_fp(z, Om, nn, beta), e2_closed(z, Om, nn, beta)
        worst = max(worst, abs(a - b) / abs(a))
print(f"  max relative diff over 400 random (Om,n,beta): {worst:.3e}")
