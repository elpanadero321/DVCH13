#!/bin/bash
# Cleanly stop the MCMC so Cobaya's signal handler checkpoints and exits.
# Then verify the chain files end on complete lines.
cd /mnt/d/DVCH13-1 || exit 1
parent=$(pgrep -f 'prterun -np 4' | head -1)
echo "sending SIGTERM to prterun ($parent) and workers"
kill -TERM "$parent" 2>/dev/null
for pid in $(pgrep -f run_dvch_cobaya_full_highl); do
  kill -TERM "$pid" 2>/dev/null
done
# give Cobaya up to 60 s to checkpoint
for _ in $(seq 1 60); do
  if ! pgrep -f run_dvch_cobaya_full_highl >/dev/null; then
    echo 'all workers stopped'
    break
  fi
  sleep 1
done
pgrep -f run_dvch_cobaya_full_highl >/dev/null && {
  echo 'still running after 60 s; sending SIGKILL'; pkill -KILL -f run_dvch_cobaya_full_highl; }
echo '=== final line counts ==='
for i in 1 2 3 4; do
  printf 'chain %s: %s lines  last-ok=%s\n' "$i" "$(wc -l < dvch_prod.$i.txt)" \
    "$(python3 -c "
import sys
try:
    line = open('dvch_prod.$i.txt','rb').readlines()[-1]
    print('yes' if len(line.split()) >= 3 else 'BAD')
except Exception as e:
    print('ERR', e)")"
done
