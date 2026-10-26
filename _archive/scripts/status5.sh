#!/bin/bash
cd /mnt/d/DVCH13-1
date +%H:%M:%S
echo "=== estado 5 cadenas ==="
ps aux | grep -a run_dvch_cobaya | grep -av grep | wc -l
wc -l dvch_prod_widebeta5.[0-9]*.txt | tail -1
tr -d '\0' < dvch_prod_widebeta5.checkpoint | grep -a -o 'Rminus1_last[^,}]*\|converged[^,}]*'
echo "=== progreso ==="
tr -d '\0' < dvch_prod_widebeta5.progress | grep -a '^[0-9]' | tail -6 | cut -c1-58
