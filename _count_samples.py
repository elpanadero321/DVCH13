import glob
for f in sorted(glob.glob("dvch_prod.*.txt")):
    n = sum(1 for l in open(f) if not l.startswith("#"))
    print(f"{f}: {n} samples")