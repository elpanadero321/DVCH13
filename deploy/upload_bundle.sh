#!/usr/bin/env bash
# ============================================================
# upload_bundle.sh — sube el bundle a un host temporal GRATIS
# y te da una URL para bajarlo desde la nube (Codespaces/Colab).
# Solo necesita curl. Sin cuentas, sin tarjeta.
#
# Uso (en WSL):
#   bash deploy/upload_bundle.sh
#   bash deploy/upload_bundle.sh /ruta/alternativa.tar.gz
# ============================================================
set -u

BUNDLE="${1:-/mnt/d/DVCH-external/dvch_portable_bundle.tar.gz}"

if [[ ! -f "$BUNDLE" ]]; then
    echo "ERROR: no existe el bundle: $BUNDLE" >&2
    exit 1
fi

SIZE="$(du -h "$BUNDLE" | cut -f1)"
NAME="$(basename "$BUNDLE")"
echo "[upload] archivo: $BUNDLE ($SIZE)"

URL=""

try() {
    local label="$1"; shift
    echo "[upload] probando $label ..."
    local out
    out="$("$@" 2>&1)" || { echo "        fallo: $(echo "$out" | tail -1)"; return 1; }
    URL="$(printf '%s\n' "$out" | grep -oE 'https?://[^[:space:]]+' | head -1)"
    if [[ -n "$URL" ]]; then
        echo "        OK -> $URL"
        return 0
    fi
    echo "        sin URL en la respuesta"
    return 1
}

# 1) bashupload.com (hasta 50 GB, sin cuenta)
try "bashupload.com" \
    curl -sS --fail -T "$BUNDLE" "https://bashupload.com/$NAME" || \
# 2) litterbox (catbox temporal, 1 GB, 72 h)
try "litterbox.catbox.moe" \
    curl -sS --fail -F "reqtype=fileupload" -F "time=72h" \
         -F "fileToUpload=@${BUNDLE}" \
         "https://litterbox.catbox.moe/resources/internals/api.php" || \
# 3) file.io (2 GB, un solo uso)
try "file.io" \
    curl -sS --fail -F "file=@${BUNDLE}" "https://file.io" || \
# 4) 0x0.st (por si se recupera)
try "0x0.st" \
    curl -sS --fail -F "file=@${BUNDLE}" "https://0x0.st" || \
# 5) transfer.sh (por si se recupera)
try "transfer.sh" \
    curl -sS --fail --upload-file "$BUNDLE" "https://transfer.sh/$NAME"

if [[ -z "$URL" ]]; then
    echo
    echo "ERROR: todos los hosts temporales fallaron." >&2
    echo "       Plan B: sube el .tar.gz a tu Google Drive y comparte el enlace," >&2
    echo "       o usa 'gh release upload' si tienes GitHub CLI." >&2
    exit 1
fi

echo
echo "============================================================"
echo "  URL DEL BUNDLE:"
echo "  $URL"
echo "============================================================"
echo
echo "En la nube (Codespace/Colab), usa esa URL asi:"
echo "  export DVCH_BUNDLE_URL='$URL'"
echo "  bash deploy/setup_cloud.sh"