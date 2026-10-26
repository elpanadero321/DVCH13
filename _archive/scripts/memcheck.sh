#!/bin/bash
echo "=== RSS por cadena cobaya (MB) ==="
ps aux | grep -a run_dvch_cobaya | grep -av grep | awk '{printf "pid=%s cpu=%s%% rss=%sMB\n", $2, $3, $6/1024}'
echo "=== CPU total disponible ==="
nproc
echo "=== RAM ==="
free -m | head -2
