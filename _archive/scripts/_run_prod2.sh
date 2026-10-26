#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1
tr -d '\r' < env.sh > /tmp/dvch_env.sh
# shellcheck disable=SC1091
source /tmp/dvch_env.sh

export DVCH_CLIK_LIB_DIR="/home/danieproyect/plc-3.1/lib"
export DVCH_MAX_SAMPLES=2000
export DVCH_BURN_IN=200
export DVCH_LEARN_PROPOSAL=true
export DVCH_COVMAT=dvch_prod_wide.covmat
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1

# Limpiar SOLO salidas previas de produccion (NO el covmat de entrada)
rm -f dvch_prod.[1-9].txt dvch_prod.progress dvch_prod.checkpoint \
      dvch_prod.updated.yaml dvch_prod.updated.dill_pickle \
      dvch_prod.input.yaml dvch_prod.input.yaml.locked

# setsid: nueva sesion desacoplada del shell que invoca el tool.
# python3 -u: sin buffer, para que el log se escriba de inmediato.
setsid bash -c "cd /mnt/d/DVCH13-1 && source /tmp/dvch_env.sh && export DVCH_CLIK_LIB_DIR=/home/danieproyect/plc-3.1/lib DVCH_MAX_SAMPLES=2000 DVCH_BURN_IN=200 DVCH_LEARN_PROPOSAL=true DVCH_COVMAT=dvch_prod_wide.covmat DVCH_CHAIN_OUTPUT=dvch_prod DVCH_CHAIN_SEED=1 && mpirun -np 4 python3 -u run_dvch_cobaya_full_highl.py" \
  > /mnt/d/DVCH13-1/dvch_prod_run.log 2>&1 < /dev/null &
disown
sleep 8
echo "=== after 8s: procs ==="
pgrep -af 'mpirun|run_dvch_cobaya_full_highl' | head
echo "=== log size ==="
wc -l dvch_prod_run.log