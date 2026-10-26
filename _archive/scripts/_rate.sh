#!/bin/bash
# Measure throughput from per-chain log lines (progress file only checkpoints
# every learn_every = 10d = 270 accepted samples/chain).
cd /mnt/d/DVCH13-1 || exit 1

snap() {
  grep -a '] Progress @' dvch_prod_run.log | tail -40 | awk '
    { chain=$1; gsub(/[\[\]:]/,"",chain); acc=$(NF-1); last[chain]=acc }
    END { s=0; for (c in last) { s+=last[c] } ; print s }'
}

a=$(snap); t0=$(date +%s)
sleep "${1:-600}"
b=$(snap); t1=$(date +%s)
python3 -c "
a,b,t0,t1 = $a,$b,$t0,$t1
dt = t1-t0
r = (b-a)*3600/dt
print(f'accepted (sum over 4 chains): {a} -> {b}  (+{b-a} in {dt}s)')
print(f'aggregate rate = {r:.0f} accepted samples/hr')
print(f'per-chain rate = {r/4:.0f} samples/hr/chain')
print(f'ETA to ~143000 total (R-1=0.03): {(143000-35138)/max(r,1e-9)/24:.1f} days')
"
