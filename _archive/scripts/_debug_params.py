h = open("dvch_calib.1.txt").readline().lstrip("#").split()
params = [c for c in h
          if c not in {"weight", "minuslogpost"}
          and not c.startswith("minuslogprior")
          and not c.startswith("chi2")]
print("n_params:", len(params))
for c in params:
    print(repr(c))