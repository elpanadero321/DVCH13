import pandas as pd

h = open("dvch_calib.1.txt").readline().lstrip("#").split()
f = pd.read_csv("dvch_calib.1.txt", sep=r"\s+", names=h, skiprows=1)
print("n_rows:", len(f))
print("total_weighted:", int(f["weight"].sum()))
print("min minuslogpost:", f["minuslogpost"].min())
print("last minuslogpost:", f["minuslogpost"].iloc[-1])
# acceptance: number of unique samples vs total
print("unique samples (weight==1):", int((f["weight"] == 1).sum()))
print("mean weight:", f["weight"].mean())