#!/bin/bash
# Lanza el check del CMB con el entorno completo (env.sh + paths).
set -eo pipefail
cd /mnt/d/DVCH13-1
source ./env.sh >/dev/null 2>&1
python3 cmb_npoint_check.py
