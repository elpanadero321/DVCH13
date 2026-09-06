#!/usr/bin/env bash
# Wrapper that sets LD_LIBRARY_PATH for the clik shared object before launching
# the Cobaya chain.  The dynamic linker reads LD_LIBRARY_PATH at process start,
# so it must be exported here rather than set from inside Python.
set -euo pipefail

export LD_LIBRARY_PATH="/home/danieproyect/plc-3.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "$(dirname "$0")"
exec python3 run_dvch_cobaya_full_highl.py "$@"