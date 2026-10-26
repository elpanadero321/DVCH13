#!/bin/bash
set -uo pipefail
cd /mnt/d/DVCH13-1 || exit 1
source env.sh >/dev/null 2>&1
echo "=== 4-proc MPI-like CAMB contention, varying OMP threads (acc=0.273) ==="
for cfg in "4 12" "4 3" "4 2" "4 1"; do
  set -- $cfg
  python3 _bench_throughput.py parallel "$1" "$2" 3
done
