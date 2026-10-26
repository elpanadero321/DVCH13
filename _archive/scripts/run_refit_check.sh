#!/bin/bash
set -eo pipefail
cd /mnt/d/DVCH13-1
source ./env.sh >/dev/null 2>&1
python3 cmb_refit_check.py
