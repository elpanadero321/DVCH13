#!/usr/bin/env python3
# Estima honestamente cuanto tardaria un run de LCDM comparado con el de DVCH,
# basado en los tiempos REALES del run de DVCH que acaba de converger.
import subprocess, datetime
subprocess.run(["bash","-lc",
  "cd /mnt/d/DVCH13-1 && tr -d '\\0' < dvch_prod_widebeta5.progress "
  "| grep -a '^[0-9]' > /tmp/p5.txt"])
rows=[]
for line in open('/tmp/p5.txt'):
    q=line.split()
    if len(q)>=4: rows.append((int(float(q[0])), q[1], float(q[3])))
if not rows: raise SystemExit("sin datos")

#Tiempo de computo NETO (suma de intervalos entre checkpoints; excluye pausas)
total_h = 0.0
prev = None
for n, ts, r in rows:
    t = datetime.datetime.fromisoformat(ts)
    if prev is not None:
        dt = (t - prev).total_seconds()/3600
        if dt < 3:   # intervalo normal de computo; >3h = pausa por apagon
            total_h += dt
    prev = t
n_final, _, r_final = rows[-1]
print(f"Run DVCH: N={n_final}, R-1={r_final:.4f}")
print(f"Tiempo de computo NETO: {total_h:.1f} horas ({total_h/24:.1f} dias)")
print(f"Tasa media: {n_final/total_h:.0f} muestras/hora")
print(f"Tiempo wall total (con pausas): "
      f"{(datetime.datetime.fromisoformat(rows[-1][1])-datetime.datetime.fromisoformat(rows[0][1])).total_seconds()/3600/24:.1f} dias")
print()
print("=== Comparacion de dimensionalidad ===")
print("DVCH (este run):  8 cosmologicos (H0,ombh2,omch2,tau,ln10As,ns,DVCH_n,DVCH_beta)")
print("                  + 18 nuisance de plik TTTEEE (gal545_A*, galf_EE_*, galf_TE_*, calib_*)")
print("                  = 26 parametros")
print()
print("LCDM (comparacion): 6 cosmologicos (H0,ombh2,omch2,tau,ln10As,ns)")
print("                     + los MISMOS 18 nuisance de plik (impuestos por la likelihood)")
print("                     = 24 parametros")
print()
print("=> La likelihood plik obliga a muestrear los 18 nuisance en CUALQUIER modelo.")
print("   LCDM solo quita 2 de 26 parametros. El coste por muestra es el mismo")
print("   (una evaluacion CAMB+clik), y la muestras necesarias para converger")
print("   son similares: 24 vs 26 dimensiones.")
print()
print("ESTIMACION REALISTA para LCDM:")
for factor, label in [(1.0, "igual"), (0.9, "10% mas rapido (menos dims)"), (0.7, "optimista")]:
    print(f"  {label}: {total_h*factor/24:.1f} dias de computo neto")
