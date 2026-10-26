#!/bin/bash
# Watchdog: snapshots cobaya state every 60s so a hard power-off loses <=1 min.
# Keeps 2 rotating backups; each copy is atomic (tmp dir + mv) so a backup is
# never left half-written even if the machine dies mid-copy.
cd /mnt/d/DVCH13-1 || exit 1
BASES=(dvch_prod_wideprior.checkpoint dvch_prod_wideprior.progress dvch_prod_wideprior.covmat)
mkdir -p ckpt_backup ckpt_backup_prev
while true; do
  STAMP=$(date +%Y-%m-%dT%H:%M:%S)
  tmp=$(mktemp -d /mnt/d/DVCH13-1/.ckpt_tmp.XXXXXX)
  cp -f "${BASES[@]}" "$tmp"/ 2>/dev/null
  cp -f dvch_prod_wideprior.[0-9]*.txt "$tmp"/ 2>/dev/null
  echo "$STAMP" > "$tmp"/snapshot_timestamp.txt
  rm -rf ckpt_backup_prev
  mv -fT ckpt_backup ckpt_backup_prev 2>/dev/null
  mv -fT "$tmp" ckpt_backup
  # last-good marker also written into the resume dir
  echo "$STAMP" > .last_good_snapshot
  sleep 60
done
