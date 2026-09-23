import importlib

for m in ("polychord", "dynesty", "getdist", "mpi4py", "numpy", "scipy", "cobaya"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
