import dill

with open("dvch_prod.updated.dill_pickle", "rb") as f:
    info = dill.load(f)
m = info["sampler"]["mcmc"]
print("PICKLE sampler:", {k: m.get(k) for k in ("learn_proposal", "Rminus1_stop",
                                                "max_samples", "burn_in")})
