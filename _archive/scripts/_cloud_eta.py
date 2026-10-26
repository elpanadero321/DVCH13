import numpy as np

t_eval = {1: 20.0, 2: 10.86, 3: 8.24, 4: 6.64}
acc = 0.27
tau = 45.0
R1 = 0.03
C = {4: 13.2, 6: 8.4, 8: 7.3, 12: 5.6, 16: 4.8, 24: 4.2, 32: 3.7}
rate_per_core = 3600.0 / (3.7 * t_eval[4]) / 4.0
print("throughput ~ %.1f accepted samples/hr per core (4 threads/chain)" % rate_per_core)
print()
print("%4s %6s %8s %9s %7s %6s" % ("m", "cores", "n_tail", "total", "hr", "days"))
for m, c in [(4, 16), (8, 32), (16, 64), (24, 96), (32, 128)]:
    n_tail = tau * C[m] / R1
    total = m * 2 * n_tail
    hr = total / (rate_per_core * c)
    print("%4d %6d %8.0f %9.0f %7.0f %6.1f" % (m, c, n_tail, total, hr, hr / 24))
