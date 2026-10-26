#!/usr/bin/env bash
# Wrapper that sets LD_LIBRARY_PATH for the clik shared object before launching
# the Cobaya chain.  The dynamic linker reads LD_LIBRARY_PATH at process start,
# so it must be exported here rather than set from inside Python.
set -euo pipefail

# Portable: read the clik lib directory from DVCH_CLIK_LIB_DIR, or fall back to
# the directory containing libclik.so via a search.  Set DVCH_CLIK_LIB_DIR in
# env.sh to point at the directory that holds libclik.so (e.g. .../plc-3.1/lib).
if [[ -n "${DVCH_CLIK_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${DVCH_CLIK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    echo "ERROR: set DVCH_CLIK_LIB_DIR (or LD_LIBRARY_PATH) to the directory" >&2
    echo "       containing libclik.so (e.g. .../plc-3.1/lib)." >&2
    exit 1
fi

cd "$(dirname "$0")"
exec python3 run_dvch_cobaya_full_highl.py "$@"