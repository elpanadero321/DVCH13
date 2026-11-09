#!/bin/bash
# PASO 0 conjunto: Planck + DESI DR1 + Pantheon+ + CC, LCDM (control) y alpha.
cd /mnt/d/DVCH13-1 || exit 9
source env.sh > /dev/null 2>&1

echo "=== JOINT 0a: LCDM control ===" 
ALPHA_MODE=lcdm ALPHA_IGNORE_PRIOR=true ALPHA_CHAIN_OUTPUT=alpha_joint_lcdm_min \
  setsid nohup python3 run_alpha_joint.py > _joint_lcdm.log 2>&1 < /dev/null &
wait

echo "=== JOINT 0b: alpha libre ==="
ALPHA_MODE=free ALPHA_IGNORE_PRIOR=true ALPHA_CHAIN_OUTPUT=alpha_joint_free_min \
  setsid nohup python3 run_alpha_joint.py > _joint_free.log 2>&1 < /dev/null &
wait

echo "JOINT_DONE"
