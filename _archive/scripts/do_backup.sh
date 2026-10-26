#!/bin/bash
# Backup de seguridad del estado actual antes del test de oversampling.
cd /mnt/d/DVCH13-1
DIR=backup_r1_0.19
mkdir -p "$DIR"
cp -f dvch_prod_widebeta.checkpoint dvch_prod_widebeta.progress \
      dvch_prod_widebeta.covmat dvch_prod_widebeta.updated.yaml \
      dvch_prod_widebeta.[0-9]*.txt "$DIR"/
echo "backup en $DIR:"
ls "$DIR" | wc -l
