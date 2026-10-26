v = open('/mnt/d/DVCH13-1/dvch_prod_wideprior.progress').read().replace('\x00','').split('\n')
rows = []
for line in v:
    p = line.split()
    if len(p) >= 4 and not p[0].startswith('#'):
        try:
            rows.append((int(float(p[0])), float(p[3])))
        except ValueError:
            pass
print("R-1 vs N (wideprior), ultimos 14:")
for n, r in rows[-14:]:
    print(f"  N={n:6d}  R-1={r:7.3f}")
