import numpy as np
from getdist import loadMCSamples

s = loadMCSamples("/mnt/d/DVCH13-1/dvch_prod", settings={"ignore_rows": 0})
names = [p.name for p in s.getParamNames().names]
idx = {n: i for i, n in enumerate(names)}
d = s.samples

print("=" * 74)
print("VEREDICTO SOBRE LA TEORIA DVCH (limite LCDM: n->0, beta->0)")
print("=" * 74)

for pname, lcdm_val, lbl in [("DVCH_n", 0.01, "n"), ("DVCH_beta", 0.0, "beta")]:
    x = d[:, idx[pname]]
    m, sd = s.mean(pname), s.std(pname)
    z = (m - lcdm_val) / sd
    q025, q975 = np.percentile(x, [2.5, 97.5])
    print(f"\n--- {lbl} ---")
    print(f"  media   = {m:.6g}")
    print(f"  sigma   = {sd:.6g}")
    print(f"  mediana = {np.median(x):.6g}")
    print(f"  IC 95%  = [{q025:.6g}, {q975:.6g}]")
    print(f"  limite LCDM = {lcdm_val}")
    print(f"  separacion del limite LCDM = {z:.2f} sigma")
    if z < 2:
        print(f"  -> NO hay deteccion significativa (consistente con LCDM a <2 sigma)")
    else:
        print(f"  -> Deteccion marginal/significativa de desviacion de LCDM")

# --- Comparacion con los valores late-time (CC+BAO) del contexto ---
print("\n" + "=" * 74)
print("CONSISTENCIA CON EL FIT LATE-TIME (CC+BAO, seccion 5.3/5.9)")
print("=" * 74)
lt_n_med, lt_n_lo, lt_n_hi = 0.2151, 0.2151 - 0.1246, 0.2151 + 0.1428
lt_b_med, lt_b_lo, lt_b_hi = 0.4953, 0.4953 - 0.3398, 0.4953 + 0.3443

n_m, n_s = s.mean("DVCH_n"), s.std("DVCH_n")
b_m, b_s = s.mean("DVCH_beta"), s.std("DVCH_beta")

print(f"\n  n  late-time = {lt_n_med:.3g}  [{lt_n_lo:.3g}, {lt_n_hi:.3g}]")
print(f"  n  CMB       = {n_m:.3g} +/- {n_s:.3g}")
print(f"  tension n    = {(n_m - lt_n_med) / np.hypot(n_s, (lt_n_hi-lt_n_lo)/2):.2f} sigma")

print(f"\n  beta late-time = {lt_b_med:.3g}  [{lt_b_lo:.3g}, {lt_b_hi:.3g}]")
print(f"  beta CMB       = {b_m:.3g} +/- {b_s:.3g}")
print(f"  PRIOR CMB beta = [0, 0.001]  <-- excluiria el valor late-time!")

dentro = (lt_b_med >= 0) and (lt_b_med <= 0.001)
print(f"  valor late-time dentro del prior CMB? {dentro}")

# --- Fraccion del posterior cerca del limite LCDM ---
print("\n" + "=" * 74)
print("MASA DEL POSTERIOR CERCA DEL LIMITE LCDM")
print("=" * 74)
xn = d[:, idx["DVCH_n"]]
frac_n = np.mean(xn < 0.03)
print(f"  P(n < 0.03)  = {frac_n:.3f}   (cerca del limite casi-LCDM)")
xb = d[:, idx["DVCH_beta"]]
frac_b = np.mean(xb < 1e-4)
print(f"  P(beta < 1e-4) = {frac_b:.3f}")
frac_both = np.mean((xn < 0.03) & (xb < 1e-4))
print(f"  P(ambos)       = {frac_both:.3f}")

print("\n" + "=" * 74)
