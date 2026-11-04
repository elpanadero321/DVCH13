#!/bin/bash
cd /mnt/d/DVCH13-1
mv revision_a_fondo.sh find_late.py recalc_slit.py fix_slit_consts.py \
   run_slit_update.py _archive/scripts/ 2>/dev/null
git add -A
git -c core.pager=cat commit -m "fix: errores anticientificos encontrados en revision final (auditoria del lenguaje)

Hallazgos graves corregidos:

1. Doble rendija con n stale (el mas peligroso): la seccion usaba n=0.22
   como 'late-time median', pero el posterior real del paper es n=0.4959.
   - dvch_double_slit.py: N_LATE 0.22 -> 0.4959, N_CMB 0.062 -> 0.063
   - F_rho recalculado: 4.1e6 -> 8.1e14 (solido, n late)
   - zeta_max: 3.8e-12 -> 1.9e-20 (n late), 2.4e-7 -> 2.3e-7 (n CMB)
   - contraste gas/oro: 2 a 8 -> 2 a ~100
   - figura dvch_double_slit.png regenerada
2. Tabla de robustez contradictoria con el CSV: 142/0.522/1.211 -> 150/0.551/1.251
   (dvch_robustness_summary.csv los decia, el tex tenia valores viejos)
3. cota de zeta inconsistente entre Sec 6 (2.4e-7) y Perspectiva futura (2.2e-7):
   unificado a 2.3e-7 (valor recalculado)
4. Afirmacion 'beta<1 satisfecho' contradecia el propio posterior beta=1.13:
   suavizado con caveat numerico explicito"
git push origin main 2>&1 | tail -1
echo "pendientes: $(git status --short | wc -l)"
