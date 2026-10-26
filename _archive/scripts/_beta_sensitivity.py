"""Verifica si DVCH_beta cambia realmente las Cls del CMB (patched CAMB)."""
import os
import sys

os.environ.setdefault("DVCH_CAMB_ROOT", "/mnt/d/DVCH-external/CAMB-master")
sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
import numpy as np
import camb

def cls(n, beta, lmax=2500):
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.3, ombh2=0.0224, omch2=0.120, tau=0.054)
    p.DVCH_flag = True
    p.DVCH_n = n
    p.DVCH_beta = beta
    p.InitPower.set_params(As=2.1e-9, ns=0.965)
    p.set_for_lmax(lmax)
    return camb.get_results(p).get_total_cls(lmax)

ref = cls(0.07, 1e-4)
lo = 2  # excluir l=0,1 (TT=0 o ruidoso)
print("Comparacion relativa de TT vs beta (n=0.07 fijo):")
print(f"{'beta':>10} {'max|dTT|/TT':>14} {'TT(2000)':>14}")
for beta in [1e-4, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.9]:
    c = cls(0.07, beta)
    rel = np.max(np.abs((c[lo:, 0] - ref[lo:, 0]) / ref[lo:, 0]))
    print(f"{beta:>10.4g} {rel:>14.3e} {c[2000,0]:>14.6e}")

print("\nAhora variando n (beta=1e-4 fijo):")
refn = cls(0.07, 1e-4)
for n in [0.01, 0.05, 0.1, 0.2, 0.25]:
    c = cls(n, 1e-4)
    rel = np.max(np.abs((c[lo:, 0] - refn[lo:, 0]) / refn[lo:, 0]))
    print(f"n={n:<6} max|dTT|/TT = {rel:.3e}")

print("\nComprobacion: el flag DVCH cambia algo vs LCDM puro?")
p0 = camb.CAMBparams()
p0.set_cosmology(H0=67.3, ombh2=0.0224, omch2=0.120, tau=0.054)
p0.InitPower.set_params(As=2.1e-9, ns=0.965)
p0.set_for_lmax(2500)
c0 = camb.get_results(p0).get_total_cls(2500)
rel = np.max(np.abs((ref[lo:, 0] - c0[lo:, 0]) / c0[lo:, 0]))
print(f"  DVCH(n=0.07,beta=1e-4) vs LCDM: max|dTT|/TT = {rel:.3e}")
