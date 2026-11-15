#!/bin/bash
# PASO 0 completo: LCDM (alpha=0, control) y luego alpha libre.
# Salida: /tmp/paso0.log + archivos .bestfit/.minimum en el repo.
cd /mnt/d/DVCH13-1 || exit 9
source env.sh > /dev/null 2>&1

echo "=== PASO 0a: LCDM control (ppf, alpha=0) ==="
bash launch_alpha_lcdm_min.sh 2>&1 | tail -20
echo
echo "=== PASO 0b: alpha libre ==="
bash launch_alpha_free_min.sh 2>&1 | tail -20
echo
echo "=== RESULTADOS ==="
for f in alpha_lcdm_min_free alpha_free_min; do
  if [ -f $f.bestfit.txt ]; then
    echo "--- $f.bestfit.txt ---"
    grep -E '^(H0|ombh2|omch2|tau|ln10As|ns|alpha|chi2__alpha_plik)' $f.bestfit.txt \
      | awk '{print $2, $1}' | head -10
  fi
done
echo "PASO0_DONE"
