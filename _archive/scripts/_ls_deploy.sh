#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1/deploy || exit 1
for f in *; do
  printf '%s\t%s bytes\n' "$f" "$(stat -c %s "$f" 2>/dev/null)"
done
EOF