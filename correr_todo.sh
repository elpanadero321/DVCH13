#!/usr/bin/env bash
# =============================================================================
# correr_todo.sh - DVCH: ejecucion turnkey de la bateria completa + MCMC Planck
# -----------------------------------------------------------------------------
# QUE HACE:
#   Corre de principio a fin TODO el pipeline real de DVCH en una maquina Linux
#   capaz (>=4 nucleos, >=16 GB RAM, internet, gfortran, MPI). No inventa ningun
#   resultado: instala, verifica, corre la bateria, corre el MCMC de Planck y
#   comprueba convergencia. Si algo falta, se detiene y te dice exactamente que.
#
# COMO USARLO (3 pasos):
#   1) Clona el repo DVCH13 y entra a su raiz (donde esta este archivo y DVCH.tex).
#   2) Pon la carpeta pkgs/ al lado con los archivos de dependencias:
#        cobaya-3.6.2.zip, class_public-master.zip,
#        COM_Likelihood_Code-v3.0_R3.01.tar.gz,
#        COM_Likelihood_Data-extra-lensing-ext_R3.00.tar.gz,
#        DataRelease-main.zip   (OBLIGATORIO para la parte CMB)
#   3) Ejecuta:   bash correr_todo.sh
#
# OPCIONES (variables de entorno):
#   DVCH_SKIP_INSTALL=1   Salta la instalacion (usala si ya instalaste antes).
#   DVCH_SKIP_CMB=1       Corre solo la parte de fondo (CC+BAO), sin Planck/MCMC.
#                         Util en plataformas sin los datos DataRelease-main.zip.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/correr_todo.log"
exec > >(tee -a "$LOG") 2>&1

line() { printf '\n============================================================\n'; }
step() { line; echo ">>> $1"; line; }
die()  { echo "ERROR: $1" >&2; echo "Revisa el log: $LOG" >&2; exit 1; }

SKIP_INSTALL="${DVCH_SKIP_INSTALL:-0}"
SKIP_CMB="${DVCH_SKIP_CMB:-0}"

# ---------------------------------------------------------------------------
step "0/6  Chequeo de prerrequisitos"
command -v python3 >/dev/null || die "Falta python3."
[ -f DVCH.tex ] || echo "AVISO: no veo DVCH.tex; asegurate de estar en la raiz del repo."
if [ "$SKIP_CMB" != "1" ]; then
  command -v gfortran >/dev/null || die "Falta gfortran (necesario para CAMB/clik). Instalalo o usa DVCH_SKIP_CMB=1."
  command -v mpirun   >/dev/null || echo "AVISO: no veo mpirun; el MCMC de Planck necesita MPI."
  [ -d pkgs ] || die "Falta la carpeta pkgs/ con las dependencias."
  [ -f pkgs/DataRelease-main.zip ] || die "Falta pkgs/DataRelease-main.zip (datos Planck high-l). Consiguelo o usa DVCH_SKIP_CMB=1."
fi
echo "Prerrequisitos OK."

# ---------------------------------------------------------------------------
if [ "$SKIP_INSTALL" = "1" ]; then
  step "1/6  Instalacion  (SALTADA por DVCH_SKIP_INSTALL=1)"
else
  step "1/6  Instalacion de dependencias (offline-friendly)"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip wheel
  [ -f requirements.txt ]     && pip install -r requirements.txt
  if [ "$SKIP_CMB" != "1" ]; then
    [ -f requirements-cmb.txt ] && pip install -r requirements-cmb.txt
    mkdir -p build data
    unzip -q -o pkgs/cobaya-3.6.2.zip -d build/           && pip install ./build/cobaya-3.6.2
    unzip -q -o pkgs/class_public-master.zip -d build/
    make -C build/class_public-master -j"$(nproc)"
    ( cd build/class_public-master/python && pip install . )
    tar xf pkgs/COM_Likelihood_Code-v3.0_R3.01.tar.gz -C build/
    ( cd build/code/plc_3.0/plc-3.01 && ./waf configure --install_all_deps && ./waf install )
    tar xf pkgs/COM_Likelihood_Data-extra-lensing-ext_R3.00.tar.gz -C data/
    unzip -q -o pkgs/DataRelease-main.zip -d data/
    echo "NOTA: compila el fork de CAMB segun README_INSTALL.md (requiere gfortran) si aun no lo hiciste."
  fi
fi

# Activa el venv si existe (por si se salto la instalacion)
if [ -d .venv ]; then source .venv/bin/activate; fi

# ---------------------------------------------------------------------------
if [ "$SKIP_CMB" != "1" ]; then
  step "2/6  Configuracion del entorno (env.sh)"
  if [ ! -f env.sh ]; then
    [ -f env.sh.example ] && cp env.sh.example env.sh || die "No hay env.sh ni env.sh.example."
    echo "Se creo env.sh a partir del ejemplo. EDITA las rutas (clik/CAMB/Planck) antes de continuar."
    die "Edita env.sh y vuelve a ejecutar (puedes usar DVCH_SKIP_INSTALL=1)."
  fi
  # shellcheck disable=SC1091
  source env.sh

  step "3/6  Preflight + smoke test de clik"
  python dvch_planck_preflight.py   || die "Preflight fallo. Revisa rutas/datos en env.sh."
  python dvch_planck_clik_smoke.py  || die "El smoke test de clik fallo."
else
  step "2-3/6  (SALTADO: modo solo-fondo, DVCH_SKIP_CMB=1)"
fi

# ---------------------------------------------------------------------------
step "4/6  Bateria de validacion (fondo, CC+BAO, crecimiento, cross-check)"
python -m pytest tests/ -q          || die "Fallaron tests unitarios."
python dvch_camb_crosscheck.py      || die "Fallo el cross-check con CAMB."
python dvch_growth_diagnostic.py    || die "Fallo el diagnostico de crecimiento/sigma8."
python dvch_joint_realdata_fit.py   || die "Fallo el ajuste conjunto CC+BAO."
python dvch_double_slit.py          || die "Fallo la prueba de congruencia doble rendija."

# ---------------------------------------------------------------------------
if [ "$SKIP_CMB" != "1" ]; then
  step "5/6  MCMC completo de Planck high-l (produccion)"
  ./launch_prod.sh                  || die "Fallo el lanzamiento del MCMC (launch_prod.sh)."
  python dvch_planck_chain_diagnostics.py || die "Diagnostico de convergencia fallo (R-hat/ESS)."
else
  step "5/6  (SALTADO: MCMC de Planck no corre en modo solo-fondo)"
fi

# ---------------------------------------------------------------------------
step "6/6  Verificacion final: los numeros del paper vs los CSV producidos"
python dvch_verify_doc_numbers.py   || die "Los numeros del manuscrito NO coinciden con los CSV. Revisa antes de reportar."

line
echo "LISTO. Todo corrio y quedo verificado."
echo "Reune los .csv / chains / logs generados y compartelos para actualizar el paper con numeros reales."
echo "Log completo: $LOG"
line
