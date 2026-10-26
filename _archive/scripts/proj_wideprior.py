import subprocess, math, datetime
subprocess.run(["bash","-lc",
  "cd /mnt/d/DVCH13-1 && tr -d '\\0' < dvch_prod_wideprior.progress "
  "| grep -a '^[0-9]' > /tmp/pw.txt"])
rows=[]
for line in open('/tmp/pw.txt'):
    q=line.split()
    if len(q)>=4 and not q[0].startswith('#'):
        try: rows.append((int(float(q[0])), q[1], float(q[3])))
        except: pass
n1,r1=rows[-1][0],rows[-1][2]
recent=[x for x in rows if x[0]>n1-8000]
if len(recent)>=2:
    t0=datetime.datetime.fromisoformat(recent[0][1]); t1=datetime.datetime.fromisoformat(recent[-1][1])
    h=(t1-t0).total_seconds()/3600
    rate=(recent[-1][0]-recent[0][0])/h
    print(f"Tasa: {rate:.0f} muestras/hora")
seg=[x for x in rows if x[0]>20000]
if len(seg)>=3:
    a=seg[0]; b=seg[-1]
    e=math.log(a[2]/b[2])/math.log(b[0]/a[0])
    print(f"Decae como N^-{e:.2f} (fase actual)")
    if e>0:
        need=n1*((r1/0.03)**(1/e))
        print(f"Para R-1<0.03: N={need:,.0f} (faltan {need-n1:,.0f})")
        if rate:
            print(f"=> {(need-n1)/rate:.0f} horas ({(need-n1)/rate/24:.1f} dias)")
