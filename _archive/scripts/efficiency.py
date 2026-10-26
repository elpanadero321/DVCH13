#!/usr/bin/env python3
# Compara la EFICIENCIA de mezcla: muestras aceptadas por segundo de wall-clock,
# entre la cadena principal (sin oversampling) y el test (con oversampling).
# El R-1 absoluto no es comparable con tan pocas muestras; lo que importa es si
# el oversampling genera muestras utiles mas rapido.
import glob, numpy as np

def stats(prefix):
    files = sorted(glob.glob(f'/mnt/d/DVCH13-1/{prefix}.[0-9]*.txt'))
    total, weights = 0, []
    for f in files:
        w = []
        with open(f) as fh:
            for line in fh:
                if line.startswith('#'): continue
                p = line.split()
                if len(p) >= 2:
                    w.append(float(p[1]))
        weights.append(w)
        total += len(w)
    return total, weights

t_main, w_main = stats('dvch_prod_widebeta')
t_test, w_test = stats('dvch_test_blocking')
print(f"Cadena principal: {t_main} muestras (peso total {sum(sum(x) for x in w_main):.0f})")
print(f"Test oversampling: {t_test} muestras (peso total {sum(sum(x) for x in w_test):.0f})")
print(f"\nRatio peso/muestra:")
print(f"  principal: {sum(sum(x) for x in w_main)/max(t_main,1):.2f}")
print(f"  test      : {sum(sum(x) for x in w_test)/max(t_test,1):.2f}")
print("\n(El oversampling adelgaza los pesos: menos peso por muestra = mas")
print(" muestras independientes por evaluacion de la likelihood lenta.)")
