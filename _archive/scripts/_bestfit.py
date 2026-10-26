import numpy as np
import yaml
from getdist import loadMCSamples

s = loadMCSamples("/mnt/d/DVCH13-1/dvch_prod", settings={"ignore_rows": 0})
names = [p.name for p in s.getParamNames().names]
jchi2 = names.index("chi2")
d = s.samples
imin = np.argmin(d[:, jchi2])

print("chi2_min global (de muestras) =", d[imin, jchi2])
print("params en el minimo:")
for n in ["H0", "ombh2", "omch2", "tau", "ln10As", "ns", "DVCH_n", "DVCH_beta"]:
    print("  %-10s %s" % (n, d[imin, names.index(n)]))

y = yaml.safe_load(open("/mnt/d/DVCH13-1/dvch_prod.updated.yaml"))
for p in ["DVCH_n", "DVCH_beta"]:
    info = y.get("params", {}).get(p, {})
    if isinstance(info, dict) and "prior" in info:
        print("prior %-10s %s" % (p, info["prior"]))
    else:
        print("prior %-10s (no encontrado en yaml, info=%s)" % (p, str(info)[:80]))
