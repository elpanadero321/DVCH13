#!/usr/bin/env bash
cd /mnt/d/DVCH13-1
echo "=== env.sh real ==="
if [ -f env.sh ]; then grep -E 'export DVCH|export LD_LIBRARY' env.sh; else echo "NO env.sh"; fi
echo "=== sample counts (lines incl header) ==="
for f in dvch_prod.1.txt dvch_prod.2.txt dvch_prod.3.txt dvch_prod.4.txt; do
  if [ -f "$f" ]; then echo "$f: $(wc -l < "$f") lines"; fi
done
echo "=== last chain output modified ==="
ls -la --time-style=long-iso dvch_prod.?.txt 2>/dev/null
echo "=== clik lib ==="
ls /home/danieproyect/plc-3.1/lib/libclik.so 2>/dev/null && echo LIBCLIK_OK || echo LIBCLIK_MISSING
echo "=== CAMB root ==="
ls -d /home/danieproyect/CAMB 2>/dev/null && ls /home/danieproyect/CAMB/fortran/DVCHModel.f90 2>/dev/null && echo CAMB_PATCHED_OK || ls -d ~/CAMB* ~/camb* 2>/dev/null
echo "=== python/cobaya in WSL ==="
python3 -c "import cobaya; print('cobaya', cobaya.__version__)" 2>&1
echo "=== any running mcmc? ==="
pgrep -af "mpirun|run_dvch" || echo "none running"