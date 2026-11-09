import sys

for path in sys.argv[1:]:
    lines = [ln for ln in open(path) if ln.strip() and not ln.startswith("#")]
    hdr = [ln for ln in open(path) if ln.startswith("#")][0].split()[1:]
    vals = lines[0].split()
    d = dict(zip(hdr, vals))
    print(f"--- {path} ---")
    for k in ["H0", "ombh2", "omch2", "tau", "ln10As", "ns", "alpha", "Msn"]:
        if k in d:
            print(f"  {k:18s} = {d.get(k)}")
    for k in d:
        if "chi2" in k:
            print(f"  {k:18s} = {d.get(k)}")
