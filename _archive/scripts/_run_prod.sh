#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1
tr -d '\r' < env.sh > /tmp/dvch_env.sh
# shellcheck disable=SC1091
source /tmp/dvch_env.sh

# Limpiar solo salidas previas de producción (NO el covmat de entrada)
rm -f dvch_prod.[1-9].txt dvch_prod.progress dvch_prod.checkpoint \
      dvch_prod.updated.yaml dvch_prod.updated.dill_pickle \
      dvch_prod.input.yaml dvch_prod.input.yaml.locked

# La colision de covmat ya esta corregida en launch_prod.sh:
#   DVCH_COVMAT=dvch_prod_wide.covmat  !=  DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CLIK_LIB_DIR="/home/danieproyect/plc-3.1/lib"
nohup bash launch_prod.sh > /tmp/launch_prod.out 2>&1 &
echo "LAUNCHED_PID=$!"
sleep 5
echo "--- launch_prod.out ---"
cat /tmp/launch_prod.out
echo "--- procs ---"
pgrep -af 'mpirun|run_dvch_cobaya_full_highl' | head