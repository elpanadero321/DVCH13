# Parche DVCH para CAMB

Este parche añade la física DVCH (interacción CDM-vacío con índice de
tracking `n` y supresión de curvatura `beta`) al CAMB estándar. Son **6
cambios** sobre un clon limpio de CAMB 2.0.4:

## 1. `fortran/DVCHModel.f90` (nuevo, reemplazable)
Módulo con:
- **Fondo** (`DVCHBackground`, `DVCHMatterFraction`): integra la
  ecuación de conservación CDM-vacío con RK4 en `u = ln(1+z)` desde
  `z=0` (donde `x = Om(z)/Om0 = 1`), guarda `ln x` en una tabla
  (paso 0.02, hasta `u = 26`) y resuelve la cerradura implícita
  `U = x^n (1+beta)/(1 + beta E2)` como raíz positiva de una cuadrática.
  La tabla se construye una vez por modelo dentro de una sección
  crítica de OpenMP (CAMB es paralelo sobre `k`); las consultas son de
  solo lectura. Salida: `dvch_x = rho_m/(rho_m0 a^-3)` (densidad de
  materia relativa a la escala estándar) y `u_frac = Om_vac/Om_vac0`.
  **Verificación** (`dvch_camb_patch_check.py` / `dvch_camb_patch_verify.py`):
  `E^2(z)` del CAMB coincide con la integración independiente (DOP853)
  a < 8e-4 para `(n, beta) = (0.2, 1e-4), (0.42, 0.53), (0.52, 1.10),
  (0.8, 1.5)`.
- **Perturbaciones** (`DVCHInteraction`): fuente de interacción para la
  ecuación del CDM (clausura de vacío en su marco de reposo).

## 2. `fortran/equations.f90`
- `dtauda` (función externa): si `DVCH_flag`, `grhov_t = grhov * U(a) * a^2`
  en vez de `DarkEnergy%BackgroundDensityAndPressure`.
- `derivs` y `derivsv` (mod `GaugeInterface`): `grhob_t`, `grhoc_t`
  multiplicadas por `dvch_x` y `grhov_t = grhov * U * a^2`, con
  `w_dark_energy_t = -1`.
- `use DVCHModel` (ya presente).

## 3. `fortran/results.f90`
- `module results` usa `DVCHModel`.
- `grho_no_de`: la materia `(grhoc + grhob)*a` se multiplica por
  `DVCHMatterFraction`.
- `CAMBdata_GetBackgroundDensities` y `DarkEnergyStressEnergy`: filas de
  materia y DE parcheadas (para que las salidas de python sean
  consistentes).

## 4. `fortran/model.f90`
Declara `DVCH_flag`, `DVCH_n`, `DVCH_beta` en `CAMBparams` (ya existente).

## 5. `fortran/Makefile_main`
`DVCHModel` en `SOURCEFILES` (ya existente).

## 6. Compilación
```bash
export FORUTILSPATH=<path a forutils>   # p. ej. /mnt/d/DVCH-external/forutils
cd fortran && make python PYCAMB_OUTPUT_DIR=../camb/
```

## Aplicación automática
`python3 _apply_camb_background_patch.py [CAMB_ROOT]` aplica los cambios
2 y 3 (idempotente) y copia `camb_patch/DVCHModel.f90`.

## Alcance y límites
- La materia que participa en la interacción es CDM **+ bariones**
  (definición de "materia" del cierre, igual que en el análisis tardío);
  los neutrinos se conservan aparte (se tratan como radiación en la
  cerradura, aproximación <0.1% en `E^2`).
- El sector de perturbaciones usa la clausura de vacío-en-su-marco de
  reposo (perturbación del vacío = 0); una tratamiento perturbativo
  completo (δQ invariante de gauge, velocidad del sonido del vacío)
  sigue siendo trabajo abierto, como dice el paper.
- Fuera del rango numérico (p. ej. `n > 0.2` con la cosmología de
  Planck a parámetros fijos) la integración de fuentes de CAMB puede no
  converger; los puntos de verificación en `cmb_npoint_profile.py` lo
  manejan.
