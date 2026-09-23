import os
import sys

egg = os.environ["DVCH_CLIK_EGG"]
print("egg path:", egg, "exists:", os.path.isdir(egg))
sys.path.insert(0, egg)
import clik

print("clik from:", clik.__file__)
print("has clik attr:", hasattr(clik, "clik"))
