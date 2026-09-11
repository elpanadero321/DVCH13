# DVCH — CONTEXTO DE SESIÓN (leer PRIMERO al abrir una conversación nueva)

> **Instrucción para la IA / el usuario:** al iniciar una conversación nueva sobre
> este proyecto, lee este archivo completo antes de hacer cualquier otra cosa.
> Contiene el estado, el entorno, los comandos, los resultados ya obtenidos y las
> trampas conocidas. Así no se repite trabajo ni se inventa nada.
> Última actualización: 2026-10-09 (auditoría: la MCMC de física completa sigue SIN ejecutar).

---

## 1. Qué es este proyecto

**DVCH** = modelo cosmológico de **interacción CDM–vacío** (dark matter–vacuum
interaction) con cerradura de Friedmann implícita. El repositorio es
`d:\DVCH13-1` y el informe LaTeX es `DVCH.tex` (compilado a `DVCH.pdf`).

Cerradura homogénea (núcleo físico, ver `dvch_full_mcmc_pipeline.py` y
`dvch_camb_background.py`):

```
E2 = Omega_r0*(1+z)^4 + Om*(1+z)^3 + OL_z
OL_z = OL * (Om_z/Om)^n * (1+beta) / (1 + beta*E2)      (punto fijo, ~30 iters)
```

Parámetros muestreados en el MCMC late-time: `Om, n, beta, H0`.
Datos: 33 Cosmic Chronometers (Moresco+2016) + 10 DESI DR1 BAO (D_M/r_d, D_V/r_d),
`r_d,fid = 147.09 Mpc`, `c = 299792.458 km/s`, `Omega_r0 = 9e-5`.

---

## 2. Entorno (VERIFICADO el 2026-09-09)

### 2.1 Windows (máquina principal)
- Working directory: `d:\DVCH13-1`
- Venv: `d:\DVCH13-1\.venv` → **Python 3.14.3** (MSC v.1944 64 bit)
- Paquetes instalados (pip list):
  `camb 2.0.4`, `cobaya 3.6.2`, `corner 2.3.0`, `emcee 3.1.6`, `getdist 1.7.7`,
  `matplotlib 3.11.1`, `mpi4py 4.1.2`, `numpy 2.5.2`, `pandas 3.0.5`,
  `PyYAML 6.0.3`, `scipy 1.18.1`
- **pytest NO está instalado** → los tests se corren con `python -m unittest`
  (el suite `tests/test_dvch.py` es `unittest.TestCase` puro, 17 tests).
- El `camb` de PyPI (2.0.4) **sí compila y corre en Windows** (wheel con
  Fortran precompilado). Sirve para cross-checks externos (ver §5.6).

### 2.2 WSL2 (Ubuntu) — stack Planck/CMB
- `wsl -l -v` → `Ubuntu` (WSL2). Estaba *Stopped*; se arranca solo al usarlo.
- Dentro de WSL (`/home/danieproyect`):
  - `plc-3.1/` con **`lib/libclik.so` OK** y **`dist/clik-3.1-py3.14-linux-x86_64.egg` OK**
  - `cobaya 3.6.2` instalado; `mpirun` y `gfortran` presentes.
  - **NO hay checkout de CAMB** (ni `/home/danieproyect/CAMB` ni `DVCHModel.f90`).
  - **NO hay archivos de datos `.clik`** (solo la librería compilada; los datos
    están bajo licencia y no están incluidos).
  - **NO existe `env.sh`** (solo `env.sh.example` en el repo).
  - `import camb` **falla** en el python de WSL.
- Consecuencia: **el MCMC Planck/CMB real (CAMB patcheado + clik) NO es
  ejecutable en este momento**. Faltan: CAMB patcheado compilado, datos `.clik`,
  y `env.sh`. Ver `README_INSTALL.md` §§3-4 y `env.sh.example` para prepararlo.
- Cadenas previas `dvch_prod.[1-4].txt` son solo *smoke* (25-42 líneas), NO
  producción convergida.

### 2.3 "Emuladores" (checkouts fuente) en `D:\Nueva carpeta (3)`
Cada carpeta contiene el repo anidado un nivel más profundo:
- `D:\Nueva carpeta (3)\bao_data-master\bao_data-master`
- `D:\Nueva carpeta (3)\CAMB-master\CAMB-master`
- `D:\Nueva carpeta (3)\class_public-master\class_public-master`
- `D:\Nueva carpeta (3)\cobaya-master\cobaya-master`
- `D:\Nueva carpeta (3)\DataRelease-main\DataRelease-main`
Son fuentes de referencia; el pipeline local NO depende de ellos (usa el `camb`
de PyPI y datos embebidos). No están compilados (CLASS/CAMB requieren gfortran).

---

## 3. Mapa de scripts (qué hace cada uno)

| Script | Función |
|---|---|
| `dvch_full_mcmc_pipeline.py` | MCMC late-time completo (emcee, CC+BAO): cadenas, resumen, R-hat/ESS, AIC/BIC, corner+traces. Defaults: 24 walkers × 2000 pasos, burn 800. |
| `dvch_mcmc_convergence.py` | Diagnóstico de convergencia multi-cadena (4 cadenas × 3000 pasos). |
| `dvch_mcmc_extended_convergence.py` | **NUEVO (esta sesión):** cadena extendida 32×8000 para verificar R-hat<1.01 sin tocar defaults. |
| `dvch_camb_crosscheck.py` | **NUEVO (esta sesión):** valida el background DVCH contra CAMB externo en el límite ΛCDM. |
| `dvch_camb_background.py` | Provider de background homogéneo DVCH (ODE + cerradura) para backends compilados. |
| `dvch_boltzmann_backend.py` | Backend de tablas de background + source DVCH. |
| `dvch_perturbations.py` | Perturbaciones sincrónicas CDM (cerradura sin fuerza CDM). |
| `dvch_relativistic_perturbation_validation.py` | Estado de validación de perturbaciones relativistas. |
| `dvch_growth_diagnostic.py` | σ8, S8, f·σ8, crecimiento. |
| `dvch_robustness_scan.py` | Scan de viabilidad del background. |
| `dvch_realdata_diagnostics.py` | Diagnósticos de datos reales (z_t, Q). |
| `dvch_joint_realdata_fit.py` | Fit conjunto DVCH vs ΛCDM con datos reales. |
| `dvch_double_slit.py` | Congruencia DVCH vs QM en doble rendija (límite de laboratorio). |
| `dvch_cmb_class_camb_mcmc.py` | Readiness CMB/CLASS/CAMB (local vs externo). |
| `dvch_planck_preflight.py` | Preflight de componentes Planck. |
| `dvch_cobaya_planck.py` | Adaptador Cobaya ↔ CAMB-DVCH ↔ clik Planck. |
| `run_dvch_cobaya_full_highl.py` | Runner MCMC Cobaya high-ℓ con priors nuisance oficiales (requiere env vars). |
| `run_dvch_cobaya_short.py` | Cadena Cobaya corta. |
| `launch_prod.sh` / `run_dvch_cobaya_full_highl.sh` | Lanzadores MPI de producción (Linux/WSL). |
| `dvch_planck_chain_diagnostics.py` | R-hat/ESS sobre cadenas Cobaya (`dvch_prod.*.txt`). |
| `tests/test_dvch.py` | 17 tests unitarios (unittest). |
| `run_full_battery.ps1` | **NUEVO (esta sesión):** batería completa secuencial con log maestro `BATTERY_MASTER.log`. |
| `camb_patch/DVCHModel.f90`, `camb_dvch_model.f90` | Módulo Fortran DVCH para parchear CAMB. |

---

## 4. Cómo correr TODO (comandos probados)

```powershell
# Batería completa (16 etapas, log maestro):
powershell -NoProfile -ExecutionPolicy Bypass -File d:\DVCH13-1\run_full_battery.ps1

# Tests unitarios (17):
cd d:\DVCH13-1; .\.venv\Scripts\python.exe -m unittest tests.test_dvch -v

# MCMC late-time completo:
.\.venv\Scripts\python.exe dvch_full_mcmc_pipeline.py

# Cross-check vs CAMB y convergencia extendida (requieren UTF-8 en consola):
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe dvch_camb_crosscheck.py
.\.venv\Scripts\python.exe dvch_mcmc_extended_convergence.py
```

Stack Planck (solo cuando esté preparado en WSL): copiar `env.sh.example`→`env.sh`,
compilar CAMB patcheado, obtener `.clik`, luego `launch_prod.sh` /
`mpirun -np 4 python3 run_dvch_cobaya_full_highl.py`.

---

## 5. Resultados de la sesión 2026-09-09 (pruebas completas)

### 5.1 Batería completa: 16/16 etapas `exit=0`
(`BATTERY_MASTER.log`, 14:43:17 → 14:52:58)

| Etapa | Duración |
|---|---|
| A_unittest_17 | 8 s (17/17 OK) |
| C1_double_slit | 6 s |
| C2_relativistic_perturb | 6 s |
| C3_cmb_readiness | 5 s |
| C4_planck_preflight | 0 s |
| C5_camb_background | 1 s |
| C6_boltzmann_backend | 1 s |
| C7_modified_solver | 10 s |
| C8_growth_sigma8 | 12 s |
| C9_robustness_scan | 26 s |
| C10_realdata_diagnostics | 6 s |
| C11_joint_realdata_fit | 100 s |
| C12_colab_demo | 11 s |
| B1_mcmc_convergence | 35 s |
| B2_full_mcmc_pipeline | 355 s |

### 5.2 Tests unitarios: **17/17 OK** (`Ran 17 tests ... OK`)

### 5.3 MCMC late-time completo (B2) — posterior
| Parámetro | mediana | media | std | q16 | q84 | q025 | q975 |
|---|---|---|---|---|---|---|---|
| Om | 0.295319 | 0.295038 | 0.018081 | 0.277658 | 0.314009 | 0.259836 | 0.329303 |
| n | 0.216394 | 0.219638 | 0.124691 | 0.082920 | 0.348025 | 0.015518 | 0.476378 |
| beta | 0.482400 | 0.488317 | 0.289912 | 0.156485 | 0.828474 | 0.029988 | 0.976562 |
| H0 | 68.375118 | 68.360107 | 1.097465 | 67.244060 | 69.476085 | 66.139844 | 70.447950 |

Convergencia (B2): R-hat = Om 1.0451, n 1.0421, beta 1.0938, H0 1.0583;
ESS = 403/458/306/551; aceptación = 0.4973.
Evidencia: n_data=43, k=4, **chi2_best=28.0279, AIC=36.0279, BIC=43.0727**.

> ⚠️ **Honestidad:** con los defaults (2000 pasos) el R-hat **NO** cumple el
> criterio del repo (R-hat<1.01); ESS sí (>100). Por eso se añadió la corrida
> extendida (§5.9) para verificar si se alcanza con más pasos. **No se debe
> afirmar convergencia R-hat<1.01 para la cadena de 2000 pasos.**

B1 (4 cadenas × 3000): max_R_hat=1.1040, min_N_eff=57.

### 5.4 Crecimiento / σ8
σ8_DVCH = 0.745420; σ8_ΛCDM ref = 0.811; ratio = 0.919137;
f·σ8(z=0): DVCH 0.438147 vs ΛCDM 0.415864.

### 5.5 Fit conjunto datos reales (123 datos)
chi2_DVCH = 96.9130; chi2_ΛCDM = 97.3010; Δchi2 = −0.3879;
ΔAIC = +1.6121; ΔBIC = +4.4243; H0_DVCH = 68.832, Om_DVCH = 0.2762, n_DVCH = 0.0715.

### 5.6 Cross-check DVCH vs CAMB externo (límite ΛCDM): **PASS**
max |ΔH|/H = **9.206e-06**, mean = 4.730e-06 (tol 1e-3).
Config: n=1e-6, beta=0, Ωm=0.30, Ωr=9e-5, H0=69.03; CAMB con
`ombh2=0.0224, omch2=0.1206, num_massive=0, omnuh2=0`.
Salidas: `dvch_camb_crosscheck.csv`, `figures/dvch_camb_crosscheck.png`.

### 5.7 Doble rendija (congruencia laboratorio)
chi2_red estándar = 0.9617989 vs DVCH = 0.9658859 (C1).
`dvch_double_slit_congruence.json`: chi2_red_QM=1.0644, chi2_red_DVCH=1.0621,
Δchi2=−0.9143, **congruent=true** (DVCH indistinguible de QM a escala de laboratorio).

### 5.8 Preflight / readiness
- Local OK: camb, cobaya, corner, emcee, getdist, matplotlib, mpi4py, numpy,
  pandas, scipy + archivos DVCH.
- **NO incluidos localmente:** `classy` (CLASS), `clik`, `plancklens`.
- Readiness CSV: clik = external; resto local.

### 5.9 Corrida extendida: **TERMINADA** (2026-09-09, ~34 min)
Script: `dvch_mcmc_extended_convergence.py` (32 walkers × 8000 pasos, burn 2000,
seed 20260909). Log: `EXT_MCMC.log` (EXIT_CODE=1 → criterio estricto no cumplido).
Salidas: `dvch_mcmc_extended_summary.csv`, `dvch_mcmc_extended_convergence.csv`,
`figures/dvch_mcmc_extended_traces.png`.

Posterior extendido (consistente con B2):
| Parámetro | mediana | + | − |
|---|---|---|---|
| Om | 0.2956 | 0.0177 | 0.0189 |
| n | 0.2151 | 0.1428 | 0.1246 |
| beta | 0.4953 | 0.3443 | 0.3398 |
| H0 | 68.3567 | 1.0930 | 1.1536 |

Convergencia extendida:
| Parámetro | R-hat | ESS | tau |
|---|---|---|---|
| Om | 1.0130 | 2724 | 70.5 |
| n | 1.0175 | 2603 | 73.8 |
| beta | 1.0137 | 1898 | 101.2 |
| H0 | 1.0138 | 2694 | 71.3 |
aceptación = 0.495.

**Veredicto honesto:** con 4× más pasos el R-hat baja de ~1.04-1.09 a
~1.013-1.018 y el ESS sube de ~300-550 a ~1900-2700, pero el criterio estricto
del repo (R-hat<1.01) queda **marginalmente sin cumplirse** (FAIL por ~0.003-0.008).
El posterior es estable y reproducible entre corridas; para R-hat<1.01 harían
falta ~2-4× más pasos o una propuesta mejor (covmat aprendida). **No afirmar
R-hat<1.01 en ninguna cadena actual.**

---

## 6. Trampas conocidas (NO repetir estos errores)

1. **Consola cp1252:** los prints con caracteres no-ASCII (Λ, á) fallan con
   `UnicodeEncodeError`. Solución: `$env:PYTHONIOENCODING='utf-8'` antes de correr.
2. **Unidades de `CAMBdata.h_of_z`:** devuelve H en **Mpc⁻¹ (c=1)**. Multiplicar
   por `299792.458` para comparar con km/s/Mpc.
3. **Neutrinos masivos por defecto en CAMB 2.0.4:** `omnuh2 ≈ 6.45e-4` (m_ν=0.06 eV).
   Para un límite ΛCDM puro hay que poner `params.num_massive=0`,
   `params.num_massless=3.046`, `params.omnuh2=0.0` (no son kwargs de `set_params`).
4. **`camb.get_background(params)`** no acepta `distances=` en v2.0.4.
5. **Scripts bash creados en Windows** quedan CRLF y bash los rechaza
   (`/dev/null\r: Permission denied`). Convertir a LF antes de correrlos en WSL.
6. **PowerShell trata stderr de python/unittest como error** (exit code 1) aunque
   el programa termine bien; leer el texto real, no el exit code.
7. **pytest no instalado** → usar `python -m unittest`.
8. **No inventar resultados:** el repo exige respaldar todo con ejecución real
   (`README_INSTALL.md` §11).

---

## 7. Pendientes / próximos pasos

1. (Hecho 2026-09-09) Resultado de la corrida extendida volcado en §5.9.
   Pendiente opcional: corrida aún más larga o covmat aprendida para R-hat<1.01.
2. Para el MCMC Planck real en WSL: (a) clonar/compilar CAMB con
   `camb_patch/` (ver `camb_patch/README_patch.md`), (b) obtener datos `.clik`
   bajo licencia, (c) crear `env.sh` desde `env.sh.example`, (d) `launch_prod.sh`.
   **Reconfirmado SIN ejecutar el 2026-10-09 — ver §9.**
3. Si se quiere R-hat<1.01 en el pipeline por defecto, aumentar `n_steps` en
   `dvch_full_mcmc_pipeline.py` (decisión de configuración, no hecha aún).
4. Compilar CLASS/CAMB desde los "emuladores" de `D:\Nueva carpeta (3)` si se
   necesita validación CLASS independiente (requiere gfortran; en WSL sí hay).

---

## 8. Inventario de salidas generadas/actualizadas esta sesión

- `BATTERY_MASTER.log` (log maestro de la batería)
- `dvch_mcmc_chains_full.csv`, `dvch_mcmc_full_summary.csv`,
  `dvch_mcmc_full_convergence.csv`, `dvch_mcmc_full_evidence.csv`
- `dvch_mcmc_convergence_summary.csv`
- `dvch_camb_crosscheck.csv`, `figures/dvch_camb_crosscheck.png`
- `dvch_sigma8_s8_summary.csv`, `dvch_growth_sigma8_table.csv`
- `dvch_joint_realdata_fit_summary.csv`, `dvch_realdata_diagnostics.csv`
- `dvch_double_slit_intensities.csv`, `dvch_double_slit_congruence.json`
- `dvch_relativistic_perturbation_status.{json,csv}`
- `dvch_cmb_class_camb_mcmc_status.csv`, `dvch_planck_preflight_report.json`
- `dvch_background_table.csv`, `dvch_boltzmann_background.csv`,
  `dvch_boltzmann_source_table.csv`, `dvch_boltzmann_modified_*.csv`
- Figuras en `figures/` (corner, traces, sigma8, robustness, double slit, etc.)
- Nuevos scripts: `run_full_battery.ps1`, `dvch_camb_crosscheck.py`,
  `dvch_mcmc_extended_convergence.py`, `_wsl_probe.sh`, `DVCH_CONTEXTO_SESION.md`,
  `run_extended.ps1`
- Corrida extendida: `EXT_MCMC.log`, `dvch_mcmc_extended_summary.csv`,
  `dvch_mcmc_extended_convergence.csv`, `figures/dvch_mcmc_extended_traces.png`

---

## 9. Auditoría 2026-10-09 — la MCMC de física completa sigue SIN ejecutar

> **Instrucción de esta sesión:** registrar el estado y dejar escrito *lo que se
> debe hacer al ejecutar*. **NO se ejecutó ningún comando de física pesada.**
> No se modificó ningún script del pipeline; solo este archivo de contexto.

### 9.1 Veredicto

**La MCMC de física completa (Planck 2018 TTTEEE+lowl+lensing + Pantheon+ + BAO
vía CAMB patcheado + clik + Cobaya/MPI) NO se ha ejecutado.** Lo único que existe
son validaciones *late-time* (CC+BAO) y cadenas *smoke* que **no convergen**.
El estado real es `prepared_not_run` / `blocked_on_external_deps`.

### 9.2 Evidencia verificada (leída en disco, no inferida)

| Fuente | Dato |
|---|---|
| `dvch_cmb_class_camb_mcmc_status.csv` | `Planck clik likelihood = False (external)` |
| `dvch_planck_preflight_report.json` (2026-09-09) | `classy = NOT_INCLUDED_LOCALLY`, `clik = NOT_INCLUDED_LOCALLY`, `plancklens = NOT_INCLUDED_LOCALLY`; `camb = OK` (PyPI, **sin parche DVCH**) |
| `dvch_planck_chain_diagnostics.csv` | R-hat de 0.96 a **13.56** (DVCH_beta R-hat=11.21) → **no convergido** |
| `dvch_prod.[1-4].txt` | 28 líneas c/u → **smoke**, no producción |
| `dvch_planck_full_highl_chain.1.txt` | 33 líneas → **smoke** |
| `dvch_planck_chain_a.1.txt` | 16 líneas → **smoke** |
| `dvch_prod_run.log` | La corrida productiva **falló**: `*ERROR* Can't open covmat file 'dvch_prod.covmat'` (MPI prun non-zero exit) |
| `env.sh` | **NO existe** (solo `env.sh.example`) |
| CAMB patcheado | **NO compilado** (existe `camb_patch/DVCHModel.f90` + `README_patch.md`, pendiente aplicar los 4 parches y compilar) |
| clik | En WSL hay `libclik.so` + egg compilados, pero **faltan los datos `.clik`** (licencia Planck) |

### 9.3 Bloqueantes exactos para la corrida real

1. **CAMB patcheado compilado** que acepte `DVCH_flag`, `DVCH_n`, `DVCH_beta`.
   Hoy solo se aplica el `camb` de PyPI (2.0.4), que **no** tiene la física DVCH.
2. **Datos Planck `.clik`** (bajo licencia) + `libclik.so` en `LD_LIBRARY_PATH`
   (la librería sí está en WSL; los datos no).
3. **`env.sh`** creado desde `env.sh.example` con las rutas reales.
4. **`dvch_prod.covmat`**: la corrida productiva murió porque el covmat de salida
   colisionaba con el de entrada. Usar `DVCH_COVMAT=dvch_prod_wide.covmat` y un
   `DVCH_CHAIN_OUTPUT` distinto.

### 9.4 LO QUE SE DEBE HACER AL EJECUTAR (NO ejecutado aún)

> Todos los comandos van dentro de **WSL2 Ubuntu**, no en PowerShell de Windows.
> Requisitos: `gfortran`, `build-essential`, `libopenmpi-dev`, `openmpi-bin`,
> `liblapack-dev`, `libfftw3-dev`.

**Paso 0 — Dependencias Python (WSL):**
```bash
pip install -r requirements-cmb.txt
```

**Paso 1 — Compilar CAMB patcheado con DVCH:**
```bash
git clone https://github.com/cmbant/CAMB.git
cd CAMB
cp /mnt/d/DVCH13-1/camb_patch/DVCHModel.f90 fortran/
# Aplicar los 4 parches manuales según camb_patch/README_patch.md:
#   - fortran/equations.f90 : use DVCHModel (x2) + bloque CDM DVCH
#   - fortran/model.f90     : DVCH_flag, DVCH_n, DVCH_beta
#   - fortran/Makefile_main : añadir DVCHModel a SOURCEFILES
#   - camb/model.py         : exponer los 3 parámetros
python setup.py build && python setup.py install
# Verificar:
python -c "import camb; p=camb.CAMBparams(); p.DVCH_flag=True; p.DVCH_n=0.2; p.DVCH_beta=1e-4; print('DVCH OK')"
```

**Paso 2 — Obtener y compilar los datos Planck clik (licencia):**
```bash
tar -xzf COM_Likelihood_Code-v3.0_R3.01.tar.gz
cd code/plc_3.0 && ./waf configure --install_all_deps && ./waf install
# Produce: plc_3.0/lib/libclik.so, plc-3.1/*.clik, clik-3.1-py3.x-*.egg
```

**Paso 3 — Crear `env.sh` desde la plantilla:**
```bash
cd /mnt/d/DVCH13-1
cp env.sh.example env.sh
# Editar env.sh con las rutas reales:
#   DVCH_CAMB_ROOT        -> /ruta/a/CAMB
#   DVCH_CLIK_EGG         -> /ruta/a/clik-3.1-py3.x-linux-x86_64.egg
#   DVCH_PLANCK_LIKELIHOOD / _LOWL / _LOWE / _LENSING -> *.clik
#   DVCH_CLIK_LIB_DIR     -> /ruta/a/plc_3.0/lib
source env.sh
```

**Paso 4 — Smoke test clik (1 evaluación, sin cadena):**
```bash
python dvch_planck_clik_smoke.py
# Valores de self-check oficiales (README_INSTALL.md §11):
#   High-l plik=-1172.47, Low-l TT commander=-11.6257,
#   Low-l EE simall=-197.99, Lensing=-4.42, A_planck=1.000442
```

**Paso 5 — Preflight (read-only):**
```bash
python dvch_planck_preflight.py   # debe mostrar clik=OK, camb=OK
```

**Paso 6 — Cadena corta (validar que arranca):**
```bash
python run_dvch_cobaya_short.py   # usa dvch_cobaya_short.yaml (12 samples)
```

**Paso 7 — Corrida productiva 4 cadenas MPI:**
```bash
# Corregir colisión de covmat (ver §9.3 punto 4) y lanzar:
export DVCH_COVMAT=dvch_prod_wide.covmat
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1
nohup mpirun -np 4 python3 run_dvch_cobaya_full_highl.py > dvch_prod_run.log 2>&1 &
# O usar el wrapper que fija LD_LIBRARY_PATH:
#   bash run_dvch_cobaya_full_highl.sh
# O el lanzador de producción:
#   bash launch_prod.sh
```

**Paso 8 — Diagnóstico de convergencia (criterio del repo):**
```bash
python3 dvch_planck_chain_diagnostics.py \
    dvch_prod.1.txt dvch_prod.2.txt dvch_prod.3.txt dvch_prod.4.txt
# Criterio: R-hat < 1.01 y ESS > 100 para TODOS los parámetros.
# Salida: dvch_planck_chain_diagnostics.csv
```

**Paso 9 — Post-proceso y evidencia:**
```bash
# Triangulares/estadísticas con GetDist sobre las cadenas convergidas.
# Registrar en este archivo SOLO resultados con ejecución real (README_INSTALL.md §11).
```

### 9.5 Notas / trampas al ejecutar

- El `camb` de PyPI **no sirve** para la corrida Planck (no tiene DVCH). Hay que
  compilar el CAMB patcheado; el PyPI solo vale para el cross-check §5.6.
- `libclik.so` debe estar en `LD_LIBRARY_PATH` **antes** de arrancar Python; los
  wrappers `run_dvch_cobaya_full_highl.sh` y `launch_prod.sh` ya lo gestionan vía
  `DVCH_CLIK_LIB_DIR`.
- No dejar que `DVCH_COVMAT` y `DVCH_CHAIN_OUTPUT` compartan nombre (causa del
  fallo de `dvch_prod_run.log`).
- El criterio de convergencia del repo es **R-hat < 1.01**; las cadenas actuales
  (`dvch_planck_chain_diagnostics.csv`, R-hat hasta 13.56) **no** lo cumplen.
- No inventar posteriores ni convergencia: todo debe respaldarse con ejecución real.

### 9.6 Estado de este archivo

- Sección añadida el **2026-10-09** como registro de auditoría.
- **No** se ejecutó ningún comando de física; **no** se tocaron scripts del pipeline.

### 9.7 Diagnóstico 2026-10-11 — la corrida SÍ arranca; muere por el covmat

> Relectura en disco de `dvch_prod_run.log` (286 líneas, corrida del 2026-09-05) y de
> `dvch_prod.progress`. **No se ejecutó física**; este apartado solo corrige/afina §9.2–9.3.

**Hallazgo principal:** la corrida productiva **no falla por dependencias externas**, sino
que **arranca, calcula física real y aborta a los ~33 s** por la colisión de covmat.

Evidencia directa del log:

| Línea(s) de `dvch_prod_run.log` | Contenido | Lectura |
|---|---|---|
| 10–17 | `plik ... got -1172.47 expected -1172.47 (diff -4.34e-07)` | clik + datos `.clik` **presentes y validados** |
| 27–37 | `Initialized external likelihood.` ×4 | CAMB-DVCH + likelihood externo OK en los 4 rangos MPI |
| 252–268 | loglikes −1831.18 / −1543.77 / −1510.53 / −1804.99; coste **12.86 s/eval** | el sampler **evalúa de verdad**, con `DVCH_n`/`DVCH_beta` muestreados |
| **26** | `From regexp 'dvch_prod[\._]covmat$' ... deleting files ['./dvch_prod.covmat']` | Cobaya **borra** el covmat de entrada |
| **273** | `*ERROR* Can't open covmat file 'dvch_prod.covmat'.` → `prun:non-zero-exit` | MPI aborta y la cadena se detiene |

**Causa raíz confirmada:** `DVCH_COVMAT` compartía nombre con el prefijo de salida
(`dvch_prod`). La regex de limpieza de Cobaya (`dvch_prod[._]covmat$`) elimina el covmat
de entrada antes de leerlo → `_load_covmat` falla → aborto MPI. Es el bloqueante §9.3-4.

**Matiz que corrige §9.2:** el log del 2026-09-05 demuestra que en esa ejecución
**sí** existían los datos Planck `.clik` (bajo `/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/`)
y **sí** se cargó clik + CAMB-DVCH con los 3 parámetros DVCH. Por tanto, para *esa*
corrida los bloqueantes 1 (CAMB parcheado) y 2 (datos `.clik`) **estaban resueltos**;
el único que mató el run fue el 4. El estado `blocked_on_external_deps` de §9.2/§9.1
puede estar desactualizado respecto a la ruta `/mnt/d/DVCH-external/planck-data/`.

**Acción aplicada (solo script de lanzamiento, sin física):** se corrigió
`launch_prod.sh` (líneas 18–23) para usar `DVCH_COVMAT=dvch_prod_wide.covmat` separado
de `DVCH_CHAIN_OUTPUT=dvch_prod`, con comentario que referencia este fallo.

**Pendiente de verificar con ejecución real (no asumido):** que tras el fix la cadena
pase de `_load_covmat`, arranque el muestreo y converja. Criterio del repo: R-hat < 1.01
y ESS > 100. No se declara convergencia hasta tener cadenas reales.



