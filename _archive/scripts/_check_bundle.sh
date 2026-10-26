#!/usr/bin/env bash
set -u
B=/mnt/d/DVCH-external/dvch_portable_bundle.tar.gz
echo "=== 1. env_remote.sh ==="
tar -tzf "$B" | grep -E 'env_remote.sh$' || echo FALTA_env_remote
echo "=== 2. covmat dentro del bundle (CRITICO) ==="
tar -tzf "$B" | grep -i 'covmat' || echo FALTA_COVMAT
echo "=== 3. planck-data: numero de entradas ==="
tar -tzf "$B" | grep -c 'planck-data/'
echo "=== 4. los 4 .clik requeridos ==="
tar -tzf "$B" | grep -E 'plik_rd12_HM_v22b_TTTEEE\.clik|commander_dx12_v3_2_29\.clik|simall_100x143_offlike5_EE_Aplanck_B\.clik|smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8\.clik_lensing' | head -8
echo "=== 5. .venv dentro del bundle (conteo) ==="
tar -tzf "$B" | grep -c '/\.venv/'
echo "=== 6. yaml/runner del bundle ==="
tar -tzf "$B" | grep -E 'dvch_cobaya_short\.yaml$|dvch_cobaya_planck\.py$|run_dvch_cobaya_full_highl\.py$'
echo "=== 7. deploy dentro del bundle (version vieja?) ==="
tar -tzf "$B" | grep -E '^dvch_bundle_stage/repo/deploy/'
echo "=== FIN ==="