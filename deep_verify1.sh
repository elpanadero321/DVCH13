#!/bin/bash
cd /mnt/d/DVCH13-1

echo "############ PARTE 1: archivos citados con texttt ############"
grep -oP 'texttt\{[A-Za-z0-9_.\\]*\}' DVCH.tex \
  | sed 's/texttt{//;s/}$//;s/\\\\/_/g' \
  | grep -E '\.(py|sh|csv|json|f90|yaml|npy|ipynb|md|tex)$' | sort -u > /tmp/cited.txt
# recuperar forma original con barras bajas
grep -o 'texttt{[A-Za-z0-9_.]*}' DVCH.tex | sed 's/texttt{//;s/}//' \
  | grep -E '\.(py|sh|csv|json|f90|yaml|npy|ipynb|md)$' | sort -u > /tmp/cited.txt
echo "archivos citados: $(wc -l < /tmp/cited.txt)"
cat /tmp/cited.txt
echo ""
echo ">>> CITADOS PERO INEXISTENTES:"
MISS=0
while read -r f; do
  HIT=$(find . -maxdepth 3 -name "$f" -not -path './_archive/*' 2>/dev/null | head -1)
  if [ -z "$HIT" ]; then
    echo "  NO EXISTE: $f"
    MISS=$((MISS+1))
  else
    echo "  ok  $f"
  fi
done < /tmp/cited.txt
echo "TOTAL FALTANTES: $MISS"
