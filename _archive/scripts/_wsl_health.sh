#!/bin/bash
echo "=== plc-3.1 ==="
ls /home/danieproyect/plc-3.1/ 2>&1 | head -3
echo "=== CAMB .so ==="
ls /mnt/d/DVCH-external/CAMB-master/camb/*.so 2>/dev/null | head -2
echo "=== espacio WSL ==="
python3 -c 'import os; s=os.statvfs("/"); print("free %.1f GB" % (s.f_bfree*s.f_bsize/1e9))'
echo "=== espacio /mnt/d ==="
python3 -c 'import os; s=os.statvfs("/mnt/d"); print("free %.1f GB" % (s.f_bfree*s.f_bsize/1e9))'
