#!/usr/bin/env bash
# ============================================================
# gcp_launch.sh — orquesta la MCMC DVCH en una VM de GCP desde
# tu WSL, por SSH. Tú creas la VM; esto hace el resto.
#
# Uso:
#   bash deploy/gcp_launch.sh <IP> <USUARIO> [ruta_clave_ssh]
#
# Ejemplo:
#   bash deploy/gcp_launch.sh 34.123.45.67 danie ~/.ssh/gcp_dvch
#
# Variables opcionales (se pasan a la VM):
#   DVCH_BUNDLE_URL   (default: litter.catbox.moe/9jijot.gz)
#   DVCH_NCHAINS=4    DVCH_MAX_SAMPLES=2000    DVCH_BURN_IN=200
#   DVCH_COVMAT=dvch_prod_wide.covmat
# ============================================================
set -euo pipefail

IP="${1:?Uso: bash deploy/gcp_launch.sh <IP> <USUARIO> [clave]}"
RUSER="${2:?Falta el usuario SSH}"
KEY="${3:-$HOME/.ssh/id_rsa}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

BUNDLE_URL="${DVCH_BUNDLE_URL:-https://litter.catbox.moe/9jijot.gz}"
NCHAINS="${DVCH_NCHAINS:-4}"
MAXS="${DVCH_MAX_SAMPLES:-2000}"
BURN="${DVCH_BURN_IN:-200}"
COVMAT="${DVCH_COVMAT:-dvch_prod_wide.covmat}"

SSHOPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
SSH=(ssh "${SSHOPTS[@]}" "$RUSER@$IP")
SCP=(scp "${SSHOPTS[@]}")

echo "=== DVCH -> GCP ==="
echo "VM      : $RUSER@$IP   (clave: $KEY)"
echo "bundle  : $BUNDLE_URL"
echo "cadenas : $NCHAINS   max_samples: $MAXS   burn_in: $BURN"
echo

echo "[1/4] comprobando conexion SSH..."
"${SSH[@]}" 'echo "  conectado a $(hostname) | cores=$(nproc) | RAM=$(free -h | awk "/^Mem:/{print \$2}")"'

echo "[2/4] instalando Docker en la VM (si falta)..."
"${SSH[@]}" 'bash -s' <<'EOSSH'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq curl ca-certificates
if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get install -y -qq docker.io
fi
sudo systemctl enable --now docker >/dev/null 2>&1 || true
echo "  docker: $(sudo docker --version)"
EOSSH

echo "[3/4] subiendo kit de despliegue + covmat..."
"${SSH[@]}" 'mkdir -p ~/deploy_kit'
"${SCP[@]}" "$HERE/gcp_remote_run.sh" "$HERE/Dockerfile" "$HERE/entrypoint.sh" "$RUSER@$IP:~/deploy_kit/"
if [[ -f "$REPO_ROOT/$COVMAT" ]]; then
    "${SCP[@]}" "$REPO_ROOT/$COVMAT" "$RUSER@$IP:~/"
    echo "  covmat subido: $COVMAT"
else
    echo "  [aviso] no encuentro $REPO_ROOT/$COVMAT (aceptacion peor)"
fi

echo "[4/4] lanzando la MCMC en la VM (desacoplada)..."
"${SSH[@]}" "DVCH_BUNDLE_URL='$BUNDLE_URL' DVCH_NCHAINS='$NCHAINS' DVCH_MAX_SAMPLES='$MAXS' DVCH_BURN_IN='$BURN' DVCH_COVMAT='$COVMAT' bash -s" <<'EOSSH'
set -e
chmod +x ~/deploy_kit/gcp_remote_run.sh
setsid nohup bash ~/deploy_kit/gcp_remote_run.sh > ~/gcp_setup.log 2>&1 < /dev/null &
echo "  setup lanzado; log: ~/gcp_setup.log"
sleep 6
tail -n 6 ~/gcp_setup.log 2>/dev/null || true
EOSSH

cat <<EOF

=== LISTO ===
La MCMC corre en segundo plano en la VM. Monitorear desde tu WSL:

  ssh -i $KEY $RUSER@$IP 'tail -f ~/dvch/dvch_prod_run.log'

Diagnostico (cuando aparezcan las cadenas):
  ssh -i $KEY $RUSER@$IP 'cd ~/dvch/dvch_bundle_stage/repo && python3 dvch_planck_chain_diagnostics.py dvch_prod.[1-4].txt'

Traer las cadenas a tu PC:
  scp -i $KEY "$RUSER@$IP:~/dvch/dvch_bundle_stage/repo/dvch_prod.*" "$REPO_ROOT/"
EOF