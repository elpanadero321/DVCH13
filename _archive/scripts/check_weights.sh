#!/bin/bash
cd /mnt/d/DVCH13-1
echo "=== pesos reales de la cadena de 4 (backup) ==="
head -5 backup_r1_0.19/dvch_prod_widebeta.1.txt | cut -c1-60
echo "..."
echo "=== aceptacion reportada (cadena 4, original) ==="
tr -d '\0' < dvch_prod_widebeta.progress | grep -a '^[0-9]' | tail -3 | cut -c1-58
echo "=== nueva cadena 5: R-1 vs N ==="
tr -d '\0' < dvch_prod_widebeta5.progress | grep -a '^[0-9]' | awk '{print $1, $4}' | tail -10
