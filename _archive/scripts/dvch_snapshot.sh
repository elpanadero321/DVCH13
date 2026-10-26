#!/bin/bash
# Snapshot periodico de seguridad para la corrida dvch_prod.
# Copia las cadenas + checkpoint + covmat al filesystem nativo de WSL (ext4),
# que esta en un vhdx con journaling distinto del drvfs /mnt/d, y rota viejos.
cd /mnt/d/DVCH13-1 || exit 1
DEST="$HOME/dvch_snapshots"
mkdir -p "$DEST"
KEEP=12
echo "[snapshot] iniciando: destino=$DEST cada 300s, keep=$KEEP"
while true; do
  TS=$(date +%Y%m%d_%H%M%S)
  TMP="$DEST/.tmp_$TS"
  rm -rf "$TMP"
  mkdir -p "$TMP"
  cp dvch_prod.1.txt dvch_prod.2.txt dvch_prod.3.txt dvch_prod.4.txt \
     dvch_prod.checkpoint dvch_prod.covmat "$TMP" 2>/dev/null || {
      echo "[snapshot] $TS error copiando; reintentando"; rm -rf "$TMP"; sleep 60; continue; }
  mv "$TMP" "$DEST/snap_$TS"
  sync
  # rotar: keep solo los KEEP mas recientes
  ls -1d "$DEST"/snap_* 2>/dev/null | sort | head -n "-$KEEP" | while read -r old; do
      rm -rf "$old"
  done
  echo "[snapshot] $TS ok ($(du -sh "$DEST/snap_$TS" | cut -f1))"
  sleep 300
done
