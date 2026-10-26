#!/bin/bash
# Limpieza segura dentro de WSL: cosas regenerables, nada del proyecto DVCH.
set -u
echo "=== antes ==="
python3 -c 'import os; s=os.statvfs("/"); print("WSL / free %.1f GB" % (s.f_bfree*s.f_bsize/1e9))'

echo "--- pip cache ---"
rm -rf ~/.cache/pip 2>/dev/null && echo "ok"
echo "--- apt cache ---"
sudo rm -rf /var/cache/apt/archives/*.deb 2>/dev/null && echo "ok (sudo)" || echo "sin sudo, omito"
echo "--- /tmp viejo (>7 dias) ---"
find /tmp -mindepth 1 -maxdepth 1 -mtime +7 -exec rm -rf {} + 2>/dev/null && echo "ok"
echo "--- logs de journal viejos ---"
sudo journalctl --vacuum-time=2d 2>/dev/null && echo "ok" || echo "sin journalctl/sudo"
echo "--- docroots de docker (imagenes ya borradas) ---"
rm -rf /var/lib/docker/overlay2 2>/dev/null && echo "ok" || echo "sin permiso (docker daemon)"

echo "=== despues ==="
python3 -c 'import os; s=os.statvfs("/"); print("WSL / free %.1f GB" % (s.f_bfree*s.f_bsize/1e9))'
