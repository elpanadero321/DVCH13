#!/usr/bin/env bash
# ============================================================
# gcp_resume.sh — sube las cadenas LOCALES a la VM de GCP y
# reanuda la MCMC desde el checkpoint (NO desde cero).
#
# Es lo que hay que correr si no puedes esperar a que termine
# localmente: transfiere los ~54 MB de cadenas + checkpoint +
# covmat, y arranca con DVCH_RESUME=true.
#
# Uso:
#   bash deploy/gcp_resume.sh <IP> <USUARIO> [clave_ssh]
#
# Ejemplo:
#   bash deploy/gcp_resume.sh 34.123.45.67 danie ~/.ssh/gcp_dvch
# ============================================================
set -euo pipefail

IP="${1:?Uso: bash deploy/gcp_resume.sh <IP> <USUARIO> [clave]}"
RUSER="${2:?Falta el usuario SSH}"
KEY="${3:-$HOME/.ssh/id_rsa}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

NCHAINS="${DVCH_NCHAINS:-4}"
MAXS="${DVCH_MAX_SAMPLES:-40000}"
R1STOP="${DVCH_RMINUS1_STOP:-0.03}"

SSHOPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
SSH=(ssh "${SSHOPTS[@]}" "$RUSER@$IP")
SCP=(scp "${SSHOPTS[@]}")

echo "=== DVCH resume -> GCP ==="
echo "VM      : $RUSER@$IP"
echo "cadenas : $NCHAINS  max_samples: $MAXS  R-1_stop: $R1STOP"
echo

# --- 0. Verificar que hay cadenas locales que subir ---
cd "$REPO_ROOT"
if ! ls dvch_prod.[1-4].txt >/dev/null 2>&1; then
    echo "ERROR: no encuentro dvch_prod.[1-4].txt en $REPO_ROOT" >&2
    exit 1
fi
NLOC=$(wc -l < dvch_prod.1.txt)
echo "[0/5] cadenas locales: ~$NLOC muestras/cadena, $(du -ch dvch_prod.[1-4].txt | tail -1 | cut -f1)"

# --- 1.Conexion + crear directorio ---
echo "[1/5] comprobando conexion SSH..."
"${SSH[@]}" 'mkdir -p ~/dvch/dvch_bundle_stage/repo' || {
    echo "ERROR: no me pude conectar a $RUSER@$IP" >&2; exit 1; }

# --- 2. Subir cadenas + checkpoint + covmat (lo que ya llevas) ---
echo "[2/5] subiendo cadenas + checkpoint + covmat (~54 MB)..."
"${SCP[@]}" dvch_prod.[1-4].txt "$RUSER@$IP:~/dvch/dvch_bundle_stage/repo/"
"${SCP[@]}" dvch_prod.checkpoint "$RUSER@$IP:~/dvch/dvch_bundle_stage/repo/" 2>/dev/null || \
    echo "  [aviso] sin checkpoint"
"${SCP[@]}" dvch_prod_learned.covmat "$RUSER@$IP:~/dvch/dvch_bundle_stage/repo/" 2>/dev/null || \
    echo "  [aviso] sin learned covmat"
"${SCP[@]}" dvch_prod_frozen.covmat "$RUSER@$IP:~/dvch/dvch_bundle_stage/repo/" 2>/dev/null || true
echo "  subido."

# --- 3. Subir el kit de despliegue ---
echo "[3/5] subiendo kit de despliegue..."
"${SSH[@]}" 'mkdir -p ~/deploy_kit'
"${SCP[@]}" "$HERE/gcp_remote_run.sh" "$HERE/Dockerfile" "$HERE/entrypoint.sh" \
    "$RUSER@$IP:~/deploy_kit/"

# --- 4. Construir la imagen (si no existe) ---
echo "[4/5] construyendo imagen docker (varios minutos, solo la primera vez)..."
"${SSH[@]}" 'bash -s' <<'EOSSH'
set -e
WORK="$HOME/dvch"
mkdir -p "$WORK/deploy"
cp -f "$HOME/deploy_kit/Dockerfile" "$WORK/Dockerfile" 2>/dev/null || true
cp -f "$HOME/deploy_kit/entrypoint.sh" "$WORK/deploy/entrypoint.sh" 2>/dev/null || true
chmod +x "$WORK/deploy/entrypoint.sh" 2>/dev/null || true
if ! sudo docker image inspect dvch-mcmc >/dev/null 2>&1; then
    echo "  [remote] construyendo imagen..."
    if [[ -d "$WORK/dvch_bundle_stage" ]]; then
        cd "$WORK" && sudo docker build -t dvch-mcmc . 2>&1 | tail -3
    else
        echo "  [remote] ERROR: falta el bundle base (dvch_bundle_stage)" >&2
        exit 1
    fi
else
    echo "  [remote] imagen dvch-mcmc ya existe"
fi
EOSSH

# --- 5. Lanzar la MCMC reanudando ---
echo "[5/5] lanzando MCMC reanudada (DVCH_RESUME=true)..."
"${SSH[@]}" "NCHAINS='$NCHAINS' MAXS='$MAXS' R1STOP='$R1STOP' bash -s" <<'EOSSH'
set -e
WORK="$HOME/dvch"
REPO="$WORK/dvch_bundle_stage/repo"
cd "$REPO"
echo "  [remote] muestras en cadena 1: $(wc -l < dvch_prod.1.txt 2>/dev/null || echo 0)"
sudo docker rm -f dvch-mcmc-run >/dev/null 2>&1 || true
setsid nohup sudo docker run --rm --name dvch-mcmc-run \
    -e DVCH_NCHAINS="$NCHAINS" \
    -e DVCH_MAX_SAMPLES="$MAXS" \
    -e DVCH_BURN_IN=0 \
    -e DVCH_RMINUS1_STOP="$R1STOP" \
    -e DVCH_LEARN_PROPOSAL=false \
    -e DVCH_COVMAT=dvch_prod_learned.covmat \
    -e DVCH_CHAIN_OUTPUT=dvch_prod \
    -e DVCH_CHAIN_SEED=1 \
    -e DVCH_RESUME=true \
    -v "$REPO:/opt/dvch/repo" \
    dvch-mcmc > "$WORK/dvch_prod_run.log" 2>&1 < /dev/null &
echo "  [remote] lanzado. Log: $WORK/dvch_prod_run.log"
sleep 8
tail -n 5 "$WORK/dvch_prod_run.log" 2>/dev/null || true
EOSSH

cat <<EOF

=== LANZADO (reanudando desde tus cadenas) ===
Monitorear desde tu WSL:

  ssh -i $KEY $RUSER@$IP 'tail -f ~/dvch/dvch_prod_run.log'

Confirmar que reanudo (no empezo de cero):

  ssh -i $KEY $RUSER@$IP 'grep -a "Found an old sample" ~/dvch/dvch_prod_run.log | tail -1'

Diagnostico cuando termine:

  ssh -i $KEY $RUSER@$IP 'cd ~/dvch/dvch_bundle_stage/repo && python3 dvch_planck_chain_diagnostics.py dvch_prod.[1-4].txt'

Traer las cadenas a tu PC:

  scp -i $KEY "$RUSER@$IP:~/dvch/dvch_bundle_stage/repo/dvch_prod.*" "$REPO_ROOT/"
EOF
