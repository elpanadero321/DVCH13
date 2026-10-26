#!/bin/bash
# Prueba de arranque de 5 cadenas: mide RAM/CPU sin llegar a muestrear.
cd /mnt/d/DVCH13-1
sed -i 's/\r$//' launch_widebeta_5chains.sh
chmod +x launch_widebeta_5chains.sh
./launch_widebeta_5chains.sh > /tmp/launch5.log 2>&1 &
sleep 100
echo "=== RAM ==="
free -m | head -2
echo "=== procesos ==="
ps aux | grep -a run_dvch_cobaya | grep -av grep | awk '{printf "rss=%dMB cpu=%s%%\n", $6/1024, $3}'
echo "=== log ==="
grep -a -E 'Initial point|Measuring|speeds|Covariance|ERROR|Error' /tmp/launch5.log | head -6
echo "=== total RAM de las 5 cadenas ==="
ps aux | grep -a run_dvch_cobaya | grep -av grep | awk '{sum+=$6} END {printf "suma RSS = %.0f MB\n", sum/1024}'
