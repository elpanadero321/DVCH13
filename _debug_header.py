h = open("dvch_calib.1.txt").readline().lstrip("#").split()
print("total tokens:", len(h))
for i, c in enumerate(h):
    print(i, repr(c))