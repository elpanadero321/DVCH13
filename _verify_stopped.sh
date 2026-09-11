#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1
echo "=== procesos dvch/mpi ==="
pgrep -af 'run_dvch|prterun|mpirun|timeout 1500' | grep -v pgrep || echo NADA
echo "=== ultimo paso registrado ==="
grep -o '[0-9]* steps taken' dvch_prod_run.log | tail -2
echo "=== cadenas escritas? ==="
ls -la dvch_prod.[1-9].txt 2>/dev/null || echo NO_HUBO_CADENAS
EOF