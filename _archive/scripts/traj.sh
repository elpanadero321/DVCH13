#!/bin/bash
cd /mnt/d/DVCH13-1
date +%H:%M:%S
echo "=== estado actual ==="
tr -d '\0' < dvch_prod_widebeta.checkpoint | grep -a -o 'Rminus1_last[^,}]*\|converged[^,}]*'
wc -l dvch_prod_widebeta.[0-9]*.txt | tail -1
echo "=== R-1 vs N (ultimos 20 puntos) ==="
tr -d '\0' < dvch_prod_widebeta.progress | grep -a '^[0-9]' | awk '{print $1, $4}' | tail -20
echo "=== tasa: muestras por hora (ultimas 6h) ==="
tr -d '\0' < dvch_prod_widebeta.progress | grep -a '^[0-9]' | tail -12 | awk '{print $1, $2}'
