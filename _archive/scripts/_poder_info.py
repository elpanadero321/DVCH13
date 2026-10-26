import numpy as np
from getdist import loadMCSamples

s = loadMCSamples("/mnt/d/DVCH13-1/dvch_prod", settings={"ignore_rows": 0})
names = [p.name for p in s.getParamNames().names]
idx = {n: i for i, n in enumerate(names)}
d = s.samples

print("=" * 74)
print("PODER INFORMATIVO DEL CMB (posterior vs prior)")
print("=" * 74)

for pname, lo, hi, lbl in [("DVCH_n", 0.01, 0.25, "n"),
                           ("DVCH_beta", 0.0, 0.001, "beta")]:
    x = d[:, idx[pname]]
    prior_sigma = (hi - lo) / np.sqrt(12.0)   # sigma de una uniforme
    post_sigma = s.std(pname)
    ratio = post_sigma / prior_sigma
    print(f"\n--- {lbl} ---")
    print(f"  sigma prior    = {prior_sigma:.6g}   (uniforme [{lo}, {hi}])")
    print(f"  sigma posterior= {post_sigma:.6g}")
    print(f"  ratio post/prior = {ratio:.3f}")
    if ratio > 0.9:
        print("  -> EL CMB NO INFORMA: posterior ~= prior (parametro no acotado)")
    elif ratio > 0.6:
        print("  -> informacion parcial (acotamiento debil)")
    else:
        print("  -> el CMB acota significativamente")

print("\n" + "=" * 74)
print("LIMITES SUPERIORES (lo que SI se puede afirmar)")
print("=" * 74)
for pname, lbl in [("DVCH_n", "n"), ("DVCH_beta", "beta")]:
    x = d[:, idx[pname]]
    q95, q99 = np.percentile(x, [95, 99])
    print(f"  {lbl:5s}: n < {q95:.5g} (95%)   n < {q99:.5g} (99%)")

print("\n" + "=" * 74)
print("CONTEXTO DEL INFORME DVCH.tex")
print("=" * 74)
print("  El informe cita 'n=0.062 (CMB-preliminary median)'.")
print(f"  Nuestra mediana CMB final = {np.median(d[:, idx['DVCH_n']]):.5g}")
print("  El informe tambien cita la doble rendija: n < 4.4e-7 (zeta=O(1)),")
print("  que excluia n=0.062 en 5 ordenes de magnitud.")
n_med = np.median(d[:, idx["DVCH_n"]])
print(f"  => nuestro n={n_med:.4g} sigue estando ~{np.log10(n_med/4.4e-7):.1f} ordenes")
print("     por encima de ese limite de laboratorio.")
