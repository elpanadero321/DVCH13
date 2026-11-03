#!/bin/bash
cd /mnt/d/DVCH13-1
cp _archive/scripts/check_tex.py . 2>/dev/null
python3 check_tex.py
mv fix_n062.py verify_posterior_claims.py verify_posterior2.py recount_chains.py _archive/scripts/ 2>/dev/null
git add -A
git -c core.pager=cat commit -m "fix: corregidos 3 errores hallados al leer las cadenas crudas

Leidas directamente dvch_prod_wideprior.[1-5].txt (no caches):
- n posterior: 0.0615 -> 0.062 (valor real de la cadena)
- fraccion n>0.45: 'below 0.1%' -> 0.11% (el valor real es 0.112%)
- 'n=0 excluido a mas de 2 sigma' -> 'excluido a ~2 sigma'
  (intervalo 95% real = [0.013, 0.256])
- La seccion mezclaba widebeta5 (117k, chi2) con wideprior (106k, posterior)
  sin distinguirlas; ahora describe los dos runs explicitamente"
git push origin main
