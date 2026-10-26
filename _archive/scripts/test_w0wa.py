import w0wa_control as w
import numpy as np

print("check E2_w0wa(0) =", w.E2_w0wa(0.0, 0.31, -1.0, 0.0), "(debe ser ~1)")
print("check w0wa limite LCDM z=1:", w.E2_w0wa(1.0, 0.31, -1.0, 0.0),
      "vs LCDM", w.m.E2_lcdm(1.0, 0.31))
print("chi2 w0wa all:", w.chi2_all_w0wa(0.31, 70.0, -19.3, -1.0, 0.0))
print("chi2 w0wa con w0=-0.9, wa=-0.5:", w.chi2_all_w0wa(0.31, 70.0, -19.3, -0.9, -0.5))
