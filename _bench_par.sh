#!/bin/bash
set -uo pipefail
cd /mnt/d/DVCH13-1 || exit 1
source env.sh >/dev/null 2>&1
echo "=== parallel contention: aggregate accepted samples/hr (acc=0.273) ==="
for cfg in "4 2" "6 2" "8 1" "12 1" "6 1" "3 4"; do
  set -- $cfg
  python3 _bench_throughput.py parallel "$1" "$2" 3
done
