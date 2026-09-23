#!/bin/bash
# Steps vs accepted over a window -> instantaneous acceptance rate.
cd /mnt/d/DVCH13-1 || exit 1

snap() {
  grep -a '] Progress @' dvch_prod_run.log | tail -40 | awk '
    { chain=$1; gsub(/[\[\]:]/,"",chain); st=$(NF-4); ac=$(NF-1)
      S[chain]=st; A[chain]=ac }
    END { s=0; a=0; for (c in A) { s+=S[c]; a+=A[c] }
          print s, a }'
}

read s0 a0 <<< "$(snap)"; t0=$(date +%s)
sleep "${1:-600}"
read s1 a1 <<< "$(snap)"; t1=$(date +%s)
python3 -c "
s0,a0,s1,a1,t0,t1 = $s0,$a0,$s1,$a1,$t0,$t1
dt = t1-t0
ds, da = s1-s0, a1-a0
print(f'steps +{ds}, accepted +{da} over {dt}s')
print(f'acceptance rate = {da/ds if ds else 0:.3f}')
print(f'aggregate accepted rate = {da*3600/dt:.0f} /hr')
"
