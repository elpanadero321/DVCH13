import numpy as np
from pathlib import Path
from _r1_diagnostic import load, sampled_cols, header_names

files = sorted(Path('.').glob('dvch_prod.*.txt'))
names = header_names(files[0])
keep = [i for i, nm in enumerate(names)
        if nm not in ("weight", "minuslogpost")
        and not nm.startswith("minuslogprior") and not nm.startswith("chi2")]
pn = [names[i] for i in keep]
chains = [sampled_cols(load(f), names)[0] for f in files]
allc = np.concatenate([c[int(len(c) * 0.5):] for c in chains], axis=0)
mu = allc.mean(0)
sd = allc.std(0)
cosmo = ['H0', 'ombh2', 'omch2', 'tau', 'ln10As', 'ns', 'DVCH_n', 'DVCH_beta']
print("cosmological params (posterior mean/std, from last 50% of current chains)")
for j, p in enumerate(pn):
    if p in cosmo:
        print("  %-10s mean=%-12.6g std=%.4g" % (p, mu[j], sd[j]))
