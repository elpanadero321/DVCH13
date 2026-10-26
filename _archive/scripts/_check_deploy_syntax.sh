#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1 || exit 1
echo "=== syntax check deploy/*.sh ==="
fail=0
for f in deploy/*.sh; do
    if tr -d '\r' < "$f" | bash -n 2>/tmp/err_$$; then
        echo "OK   : $f"
    else
        echo "FAIL : $f"
        cat /tmp/err_$$
        fail=1
    fi
done
echo "=== deploy/ contenido ==="
ls -1 deploy/
echo "=== exit_status=$fail ==="
EOF