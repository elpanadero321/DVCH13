#!/usr/bin/env bash
# Sonda del stack Planck/CAMB/clik dentro de WSL (solo lectura).
echo "--- HOME listing ---"
ls -la /home/danieproyect/ 2>/dev/null | head -40
echo "--- CAMB candidates ---"
find /home/danieproyect -maxdepth 3 \( -iname '*camb*' -o -iname 'DVCHModel.f90' \) 2>/dev/null | head -20
echo "--- clik data (.clik) ---"
find /home/danieproyect -maxdepth 5 -name '*.clik' 2>/dev/null | head -10
echo "--- lensing ---"
find /home/danieproyect -maxdepth 5 -name '*.clik_lensing' 2>/dev/null | head -5
echo "--- clik egg ---"
find /home/danieproyect -maxdepth 5 -name 'clik-3.1-*.egg' 2>/dev/null | head -5
echo "--- plc dirs ---"
ls -d /home/danieproyect/plc* 2>/dev/null
echo "--- python & mpi ---"
which python3 mpirun gfortran 2>/dev/null
python3 -c "import camb, sys; print('camb', camb.__version__, sys.executable)" 2>&1 | head -3
