import os
import sys

os.environ.setdefault("DVCH_CAMB_ROOT", "/mnt/d/DVCH-external/CAMB-master")
os.environ.setdefault("DVCH_CLIK_EGG", "/home/danieproyect/plc-3.1/clik-egg")
os.environ.setdefault("DVCH_CLIK_LIB_DIR", "/home/danieproyect/plc-3.1/lib")
os.environ["LD_LIBRARY_PATH"] = os.environ["DVCH_CLIK_LIB_DIR"]
sys.path.insert(0, "/mnt/d/DVCH13-1")

import numpy as np

# Reproducir la carga de covmat de cobaya
path = "/mnt/d/DVCH13-1/dvch_prod_widebeta.covmat"
with open(path) as f:
    header = f.readline()
names = header.strip().lstrip("#").split()
print(f"nombres en cabecera: {len(names)}")
print("primeros:", names[:8])

body = np.loadtxt(path)
print(f"matriz: {body.shape}")
assert body.shape == (len(names), len(names)), "dimensiones no cuadran"

# Verificar simetria y PD
sym = np.allclose(body, body.T, atol=1e-10)
ev = np.linalg.eigvalsh(body)
print(f"simetrica: {sym}")
print(f"autovalor min: {ev.min():.3e}")
print(f"autovalor max: {ev.max():.3e}")
print(f"condicion: {ev.max()/max(ev.min(),1e-30):.3e}")
if not sym:
    d = np.abs(body - body.T)
    print(f"  asimetria maxima: {d.max():.3e} en {np.unravel_index(d.argmax(), d.shape)}")
