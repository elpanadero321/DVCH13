import math
data = []
for line in open('/tmp/prog.txt'):
    p = line.split()
    if len(p) >= 3:
        data.append((int(float(p[0])), p[1], float(p[2])))
n1, r1 = data[-1][0], data[-1][2]
print(f"N actual = {n1}, R-1 actual = {r1:.4f}")
falta = r1 / 0.03
print(f"Falta un factor {falta:.1f} para R-1 < 0.03\n")

def decay(seg):
    n0, r0 = seg[0][0], seg[0][2]
    n1b, r1b = seg[-1][0], seg[-1][2]
    if r1b >= r0: return None
    return math.log(r0 / r1b) / math.log(n1b / n0)

for label, seg in [
    ("largo plazo (N>11000)", [d for d in data if d[0] > 11000]),
    ("ultimas 10k muestras", [d for d in data if d[0] > n1 - 10000]),
    ("ultimas 5k muestras", [d for d in data if d[0] > n1 - 5000]),
]:
    e = decay(seg)
    if e is None:
        print(f"{label}: R-1 NO esta bajando"); continue
    mult = falta ** (1.0 / e)
    n_need = n1 * mult
    print(f"{label}: decae como N^-{e:.2f}")
    print(f"   => faltarian {n_need - n1:,.0f} muestras = {(n_need-n1)/440/24:.1f} dias a 440/hora")
print(f"\nmax_samples = 200000, llegaria en {(200000-n1)/440/24:.1f} dias")
