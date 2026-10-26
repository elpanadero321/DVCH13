#!/bin/bash
echo "=== proceso LCDM minimize ==="
ps aux | grep -a run_dvch_cobaya_lcdm | grep -av grep | awk '{print "pid="$2, "cpu="$3"%", "tiempo="$10}'
echo "=== archivos de salida ==="
ls -la /mnt/d/DVCH13-1/ | grep -a 'lcdm_min.minimum'
echo "=== log cobaya si existe ==="
ls /mnt/d/DVCH13-1/dvch_lcdm_min* 2>/dev/null
