#!/bin/bash
cd /mnt/d/DVCH13-1
mv revision_a_fondo.sh _archive/scripts/ 2>/dev/null
git add -A
git -c core.pager=cat commit -m "fix: revision a fondo - unifica valores y corrige errores menores

- n posterior del CMB unificado a 0.063 en todo el manuscrito
  (antes 0.0615/0.062/0.063 mezclados)
- beta unificado a 1.00 [0.35, 1.65]
- tests: '17 automated' -> '21 automated' (el conteo real)
- typo 'release-time' -> 'real-data'; 'a Confrontation' -> 'a confrontation'
- conteo de muestras: '1.1x10^5' -> '1.17x10^5' (consistente)"
git push origin main 2>&1 | tail -1
echo "pendientes: $(git status --short | wc -l)"
