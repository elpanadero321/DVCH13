import os
import sys

sys.path.insert(0, "/mnt/d/DVCH13-1")
os.chdir("/mnt/d/DVCH13-1")
os.environ["ALPHA_MODE"] = "lcdm"
from run_alpha_joint import build_info, MODE  # noqa: E402

print("MODE =", MODE)
info = build_info()
print("params:", sorted(info["params"].keys()))
lk = info["likelihood"]["alpha_joint"]
print("external:", lk["external"])
print("sig:", [p for p in lk["external"].__signature__.parameters])
