#!/usr/bin/env bash
# ============================================================
# gcp_preflight.sh — verifica TODO antes de gastar credito.
# Corre ESTO primero. Falla sin costo si algo no cierra.
#
# Uso:
#   bash deploy/gcp_preflight.sh <IP> <USUARIO> [clave_ssh]
# ============================================================
set -euo pipefail

IP="${1:?Uso: bash deploy/gcp_preflight.sh <IP> <USUARIO> [clave]}"
RUSER="${2:?Falta el usuario SSH}"
KEY="${3:-$HOME/.ssh/id_rsa}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

SSHOPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
SSH=(ssh "${SSHOPTS[@]}" "$RUSER@$IP")

echo "=== PRE-FLIGHT GCP (sin costo) ==="
echo

# --- 1. Conexion + capacidad de la VM ---
echo "[1/6] capacidad de la VM..."
"${SSH[@]}" 'bash -s' <<'EOSSH'
echo "  host      : $(hostname)"
echo "  cores     : $(nproc)"
echo "  RAM       : $(free -h | awk "/^Mem:/{print \$2}")"
echo "  RAM libre : $(free -h | awk "/^Mem:/{print \$4}")"
echo "  disco     : $(df -h "$HOME" | awk "NR==2{print \$4\" libres\"})"
EOSSH
echo

# --- 2. Que las cadenas locales esten completas ---
echo "[2/6] cadenas locales..."
cd "$REPO_ROOT"
for f in dvch_prod.1.txt dvch_prod.2.txt dvch_prod.3.txt dvch_prod.4.txt; do
    if [[ ! -s "$f" ]]; then
        echo "  FALTA: $f"
        exit 1
    fi
    echo "  $f : $(wc -l < "$f") lineas"
done
echo "  checkpoint : $(test -s dvch_prod.checkpoint && echo OK || echo FALTA)"
echo "  covmat     : $(test -s dvch_prod_learned.covmat && echo OK || echo FALTA)"
echo

# --- 3. Que Docker funcione en la VM ---
echo "[3/6] docker en la VM..."
"${SSH[@]}" 'bash -s' <<'EOSSH'
if command -v docker >/dev/null 2>&1; then
    echo "  docker: $(docker --version)"
elif command -v sudo >/dev/null 2>&1 && sudo -n docker version >/dev/null 2>&1; then
    echo "  docker: $(sudo docker --version)"
else
    echo "  docker: NO instalado (se instalara en el arranque)"
fi
EOSSH
echo

# --- 4. Estimacion de costo y tiempo ---
echo "[4/6] estimacion de recurso restante..."
NLOC=$(wc -l < dvch_prod.1.txt)
REMAIN=$(( 40000 - NLOC ))
"${SSH[@]}" "bash -s" <<EOSSH
CORES=\$(nproc)
# ~76 muestras/h/cadena local a 12 cores; escala con cores
RATE=\$(echo "scale=1; 76 * $CORES / 12" | bc 2>/dev/null || echo "n/a")
HOURS=\$(echo "scale=1; $REMAIN / $RATE" | bc 2>/dev/null || echo "n/a")
echo "  muestras restantes/cadena : $REMAIN"
echo "  cores VM                  : \$CORES"
echo "  muestras/h/cadena (estim) : \$RATE"
echo "  horas de reloj (estim)    : \$HOURS"
EOSSH
echo

# --- 5. Verificar que el bundle base existe o se puede bajar ---
echo "[5/6] bundle base en la VM..."
"${SSH[@]}" 'bash -s' <<'EOSSH'
WORK="$HOME/dvch"
if [[ -d "$WORK/dvch_bundle_stage" ]]; then
    echo "  bundle presente: $(du -sh "$WORK/dvch_bundle_stage" | cut -f1)"
else
    echo "  bundle ausente -> gcp_resume.sh necesitara el bundle base"
    echo "  (corre antes deploy/gcp_launch.sh, o sube el bundle a mano)"
fi
EOSSH
echo

# --- 6. Resumen go/no-go ---
echo "[6/6] RESUMEN"
echo "  cadenas a subir : $(du -ch dvch_prod.[1-4].txt dvch_prod.checkpoint dvch_prod_learned.covmat 2>/dev/null | tail -1 | cut -f1)"
echo
echo "  Si todo arriba dice OK o 'se instalara', lanza:"
echo "    bash deploy/gcp_resume.sh $IP $RUSER ${KEY}"
echo
echo "  Si algo FALLO: no gastes nada. Arregla lo local primero."
