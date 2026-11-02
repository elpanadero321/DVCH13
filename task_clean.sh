#!/bin/bash
cd /mnt/d/DVCH13-1
for f in deep_verify1.sh final_tidy.sh check_tex.py verify_numbers.py; do
  if [ -f "$f" ]; then
    mv "$f" _archive/scripts/
    echo "archivado: $f"
  else
    echo "no existe: $f"
  fi
done
echo "---"
git status --short | head -10
