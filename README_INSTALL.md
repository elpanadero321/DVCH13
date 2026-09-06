# DVCH Pipeline — Instalación y Ejecución Portable

Este documento explica cómo instalar y ejecutar el pipeline DVCH (background + perturbaciones + CAMB patcheado + likelihoods oficiales de Planck + Cobaya MCMC) en **cualquier máquina Linux** (nativa o WSL2).

> **Importante:** El pipeline requiere los datos oficiales de Planck (clik), que están bajo licencia. Debes obtenerlos del repositorio oficial de Planck (https://pla.esac.esa.int o el tarball `COM_Likelihood_Code-v3.0_R3.01.tar.gz`). Este paquete NO incluye los datos clik por licencia.

---

## 1. Requisitos del sistema

- **Linux** (nativo o WSL2). Se recomienda WSL2 con Ubuntu 22.04+.
- **Python 3.10+** (probado con 3.14).
- **Compilador Fortran** (`gfortran`) para compilar CAMB.
- **OpenMPI** (`mpirun`) para las cadenas MCMC paralelas.
- **≥ 8 GB RAM**, **≥ 8 cores** (recomendado 12+).
- **~2 GB de disco** (CAMB + clik + datos Planck).

Instalar dependencias del sistema (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y gfortran build-essential python3 python3-pip \
    libopenmpi-dev openmpi-bin liblapack-dev libfftw3-dev
```

---

## 2. Instalar dependencias Python

```bash
pip install numpy scipy pandas matplotlib emcee corner \
    camb cobaya getdist mpi4py pyyaml
```

> **Nota sobre CAMB:** El pipeline usa un CAMB **patcheado** con la física DVCH (ver sección 4). No uses el CAMB de PyPI directamente; compila el CAMB patcheado.

---

## 3. Obtener los datos oficiales de Planck (clik)

1. Descarga el código de likelihood de Planck 2018:
   - `COM_Likelihood_Code-v3.0_R3.01.tar.gz` desde el Planck Legacy Archive.
2. Extrae y compila clik:

```bash
tar -xzf COM_Likelihood_Code-v3.0_R3.01.tar.gz
cd code/plc_3.0
./waf configure --install_all_deps
./waf install
```

3. Esto produce:
   - `plc_3.0/lib/libclik.so` (la librería compartida)
   - `plc_3.0/plc-3.1/` (los datos .clik)
   - El egg de Python `clik-3.1-py3.x-linux-x86_64.egg`

4. Anota las rutas:
   - `CLIK_LIB_DIR` = directorio que contiene `libclik.so` (ej. `.../plc_3.0/lib`)
   - `CLIK_EGG` = ruta al egg (ej. `.../clik-3.1-py3.14-linux-x86_64.egg`)
   - `PLANCK_DATA` = directorio con los `.clik` (ej. `.../plc_3.0/plc-3.1/`)

---

## 4. Compilar el CAMB patcheado con DVCH

El pipeline usa un CAMB modificado que añade la física DVCH (interacción CDM-vacío). El parche consiste en:

1. **`fortran/DVCHModel.f90`** — nuevo módulo (incluido en este paquete).
2. **`fortran/equations.f90`** — añade la llamada a `DVCHInteraction` en la ecuación de movimiento del CDM.
3. **`fortran/model.f90`** — añade los parámetros `DVCH_flag`, `DVCH_n`, `DVCH_beta`.
4. **`fortran/Makefile_main`** — añade `DVCHModel` a `SOURCEFILES`.
5. **`camb/model.py`** — expone los 3 parámetros a Python.

### Pasos:

```bash
# 1. Clonar CAMB
git clone https://github.com/cmbant/CAMB.git
cd CAMB

# 2. Copiar el módulo DVCH
cp /ruta/al/paquete/camb_patch/DVCHModel.f90 fortran/

# 3. Aplicar los parches (ver camb_patch/README_patch.md)
#    - equations.f90: añadir el bloque DVCH en la ecuación CDM
#    - model.f90: añadir los 3 parámetros
#    - Makefile_main: añadir DVCHModel a SOURCEFILES
#    - camb/model.py: añadir los 3 parámetros

# 4. Compilar
python setup.py build
python setup.py install
# o en modo desarrollo:
pip install -e .
```

> **Verificación:** Después de compilar, `python -c "import camb; print(camb.__version__)"` debe funcionar, y `camb.set_params(DVCH_flag=True, DVCH_n=0.2, DVCH_beta=1e-4)` no debe dar error.

---

## 5. Configurar las rutas

El pipeline usa variables de entorno para localizar los recursos. Crea un archivo `env.sh`:

```bash
#!/usr/bin/env bash
# Ajusta estas rutas a tu máquina
export DVCH_CAMB_ROOT="/ruta/a/CAMB"                    # directorio raíz de CAMB patcheado
export DVCH_CLIK_EGG="/ruta/a/clik-3.1-py3.14-linux-x86_64.egg"
export DVCH_PLANCK_LIKELIHOOD="/ruta/a/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"
export DVCH_PLANCK_LOWL="/ruta/a/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik"
export DVCH_PLANCK_LOWE="/ruta/a/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"
export DVCH_PLANCK_LENSING="/ruta/a/plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing"
export LD_LIBRARY_PATH="/ruta/a/plc_3.0/lib:${LD_LIBRARY_PATH:-}"
```

> **CRÍTICO:** `LD_LIBRARY_PATH` debe apuntar al directorio con `libclik.so` y debe exportarse **antes** de lanzar Python (el linker dinámico lo lee al inicio del proceso).

---

## 6. Ejecutar las pruebas

```bash
source env.sh
cd /ruta/al/paquete

# 1. Pruebas unitarias locales (17 tests)
python -m pytest tests/test_dvch.py -v

# 2. Smoke test de clik (verifica que los likelihoods cargan)
python dvch_planck_clik_smoke.py

# 3. Preflight (verifica rutas y datos)
python dvch_planck_preflight.py
```

---

## 7. Ejecutar la cadena MCMC de producción

```bash
source env.sh
cd /ruta/al/paquete

# Configuración de la cadena (ajusta según tu hardware)
export DVCH_MAX_SAMPLES=400      # muestras aceptadas por cadena
export DVCH_BURN_IN=200          # burn-in por cadena
export DVCH_LEARN_PROPOSAL=false
export DVCH_RMINUS1_STOP=0.01
export DVCH_COVMAT=dvch_prod_wide.covmat
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=42

# Lanzar 4 cadenas MPI
mpirun -np 4 python3 run_dvch_cobaya_full_highl.py
```

> **Nota sobre el covmat:** `dvch_prod_wide.covmat` es una matriz de covarianza inicial (26×26) ensanchada por factor 5. Si tu máquina tiene más cores, puedes aumentar `-np` (pero cada cadena usa ~2 cores por OpenMP de CAMB).

---

## 8. Calcular convergencia (R-hat / ESS)

Después de que las cadenas terminen:

```bash
python3 dvch_planck_chain_diagnostics.py \
    dvch_prod.1.txt dvch_prod.2.txt dvch_prod.3.txt dvch_prod.4.txt
```

Esto produce `dvch_planck_chain_diagnostics.csv` con R-hat (Gelman-Rubin) y ESS por parámetro.

**Criterio de convergencia:** R-hat < 1.01 y ESS > 100 para todos los parámetros.

---

## 9. Estructura del paquete

```
DVCH13-1/
├── README_INSTALL.md          # este archivo
├── env.sh.example             # plantilla de variables de entorno
├── camb_patch/                # parche DVCH para CAMB
│   ├── DVCHModel.f90          # módulo Fortran nuevo
│   └── README_patch.md        # instrucciones de parcheo
├── run_dvch_cobaya_full_highl.py   # runner principal MCMC
├── dvch_cobaya_planck.py      # adaptador CAMB-DVCH/Planck
├── dvch_cobaya_short.yaml     # config base Cobaya
├── dvch_planck_chain_diagnostics.py  # R-hat/ESS
├── dvch_planck_clik_smoke.py  # smoke test clik
├── dvch_planck_preflight.py   # preflight
├── dvch_prod_wide.covmat      # covmat inicial ensanchada
├── dvch_background_table.csv  # tablas de background
├── dvch_boltzmann_backend.py  # backend Boltzmann
├── dvch_perturbations.py      # perturbaciones
├── dvch_camb_background.py    # background CAMB
├── dvch_full_mcmc_pipeline.py # pipeline MCMC completo
├── dvch_mcmc_convergence.py   # convergencia MCMC
├── tests/test_dvch.py         # 17 tests unitarios
└── DVCH.tex                   # reporte LaTeX
```

---

## 10. Solución de problemas

| Problema | Causa | Solución |
|----------|-------|----------|
| `ImportError: libclik.so` | `LD_LIBRARY_PATH` no apunta a `libclik.so` | `export LD_LIBRARY_PATH=/ruta/a/plc_3.0/lib` antes de Python |
| `Can't open covmat file` | El covmat de entrada tiene el mismo nombre que el output | Renombrar el covmat de entrada (ej. `dvch_prod_wide.covmat`) |
| Aceptación MCMC < 1% | Covmat demasiado estrecha | Usar `dvch_prod_wide.covmat` (scale=5) o ensanchar más |
| `mpirun: command not found` | OpenMPI no instalado | `sudo apt install openmpi-bin libopenmpi-dev` |
| CAMB no reconoce `DVCH_flag` | CAMB no patcheado | Recompilar CAMB con el parche (sección 4) |
| WSL2 se reinicia y mata la corrida | Windows suspende | `powercfg /change standby-timeout-ac 0` en Windows |

---

## 11. Notas sobre reproducibilidad

- **Nunca inventar priors, espectros, posteriores o convergencia.** Todo debe respaldarse con ejecución real.
- Los priors nuisance oficiales están en `run_dvch_cobaya_full_highl.py` (`OFFICIAL_PRIORS`).
- Los valores de self-check oficiales: High-ℓ plik = -1172.47, Low-ℓ TT commander = -11.6257, Low-ℓ EE simall = -197.99, Lensing = -4.42, A_planck = 1.000442.
- La convergencia estadística (R-hat < 1.01) debe demostrarse con cadenas reales, no asumirse.