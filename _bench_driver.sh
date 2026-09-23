#!/bin/bash
# Driver: CAMB thread scaling + clik cost + parallel contention on 12 cores.
set -uo pipefail
cd /mnt/d/DVCH13-1 || exit 1
source env.sh >/dev/null 2>&1

echo "=== single-proc CAMB scaling ==="
for T in 1 2 3 4; do
  OMP_NUM_THREADS=$T python3 _bench_throughput.py camb 3
done

echo "=== clik (serial) ==="
python3 _bench_throughput.py clik

echo "=== parallel contention: aggregate accepted samples/hr (acc=0.273) ==="
for cfg in "4 2" "6 2" "8 1" "12 1" "6 1" "3 4"; do
  set -- $cfg
  python3 _bench_throughput.py parallel "$1" "$2" 3
done
