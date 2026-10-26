#!/usr/bin/env python3
# Proyeccion de la cadena de 5 con los datos REALES actuales.
import subprocess, math
subprocess.run(["bash","-lc",
  "cd /mnt/d/DVCH13-1 && tr -d '\\0' < dvch_prod_widebeta5.progress "
  "| grep -a '^[0-9]' > /tmp/p5.txt"])
d=[]
for line in open('/tmp/p5.txt'):
    q=line.split()
    if len(q)>=4: d.append((int(float(q[0])), float(q[3])))
n1,r1=d[-1]
print(f"Actual: N={n1}, R-1={r1:.3f}, objetivo R-1<0.03\n")
# tasa de muestras/hora con timestamps
d2=[]
for line in open('/tmp/p5.txt'):
    q=line.split()
    if len(q)>=4: d2.append((int(float(q[0])), q[1], float(q[3])))
import datetime
tail=[x for x in d2 if x[0]>n1-4000]
if len(tail)>=2:
    t0=datetime.datetime.fromisoformat(tail[0][1]); t1=datetime.datetime.fromisoformat(tail[-1][1])
    h=(t1-t0).total_seconds()/3600
    rate=(tail[-1][0]-tail[0][0])/h
    print(f"Tasa: {rate:.0f} muestras/hora")
# decaimiento en la fase actual (R-1 < 10)
seg=[x for x in d if x[1]<10 and x[0]>1000]
if len(seg)>=3:
    a=seg[0]; b=seg[-1]
    e=math.log(a[1]/b[1])/math.log(b[0]/a[0])
    print(f"Decae como N^-{e:.2f} en la fase actual")
    if e>0:
        need=n1*((r1/0.03)**(1/e))
        print(f"Para R-1<0.03: N={need:,.0f} (faltan {need-n1:,.0f})")
        if 'rate' in dir():
            print(f"=> {(need-n1)/rate:.1f} horas ({(need-n1)/rate/24:.1f} dias)")
# comparar con regimen asintotico N^-0.5
print(f"\nEscenario pesimista (N^-0.5): N={n1*((r1/0.03)**2):,.0f}")
if 'rate' in dir():
    print(f"=> {((n1*((r1/0.03)**2))-n1)/rate/24:.1f} dias")
