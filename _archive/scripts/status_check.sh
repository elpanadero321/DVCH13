#!/bin/bash
echo "=== todos los procesos python/mpirun ==="
ps aux | grep -aE 'python|mpirun|cobaya' | grep -av grep
echo "=== log completo reciente ==="
cd /mnt/d/DVCH13-1
ls -t resume_*.log | head -1 | xargs wc -l
ls -t resume_*.log | head -1 | xargs tail -15
