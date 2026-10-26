import sys
print(sys.path[:6])
try:
    import clik
    print("clik from:", clik.__file__)
    print("has clik attr:", hasattr(clik, "clik"))
except Exception as e:
    print("import error:", e)
