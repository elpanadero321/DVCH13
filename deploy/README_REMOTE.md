# Correr la MCMC DVCH completa en OTRA máquina (sin recortes)

Este directorio contiene todo lo necesario para mover la corrida
**definitiva** (fase 3: covmat óptimo congelado, `learn_proposal=false`,
`Rminus1_stop=0.03`, refs = medias posteriores reales) a un servidor / HPC /
VM de nube, sin tocar los parámetros científicos.

## Qué cambió respecto a la versión anterior

La versión antigua lanzaba una corrida de juguete (4×2000, covmat viejo
`dvch_prod_wide.covmat`, `learn_proposal=true`). Ahora el bundle lleva:

- **`dvch_prod_frozen.covmat`**: la propuesta aprendida y verificada
  (coincide con la covarianza posterior real a pocos %). Nombre seguro: no
  coincide con el regexp de limpieza de Cobaya `dvch_prod[._]covmat$`, así
  sobrevive a un arranque limpio.
- **refs = medias posteriores** en `dvch_cobaya_short.yaml` (H0=67.28,
  omch2=0.1202, …): las cadenas arrancan dentro del conjunto típico y el
  burn-in es mínimo.
- **`Rminus1_stop=0.03`** (no 0.01): R−1<0.01 costaría ~33 días en este
  hardware; 0.03 da ESS≈1400/param en una fracción del tiempo. Ver
  `_r1_diagnostic.py` y `_proposal_check.py`.
- **`learn_proposal=false`**: la propuesta ya es óptima, así la fase remota es
  una cadena de Markov limpia y homogénea (sin más adaptación).

## Por qué esto y no "ejecutarlo en la nube desde aquí"

No existe forma de que un asistente lance el cálculo en un servidor que no
controla. El cómputo debe correr en una máquina a la que **tú** tengas acceso.
Lo que se hace aquí es **empaquetar y automatizar** ese traslado.

## Paso 1 — Empaquetar (en tu WSL)

```bash
bash deploy/package_dvch.sh
```

Produce `dvch_portable_bundle.tar.gz` con:
- el repo DVCH (sin salidas),
- **CAMB ya parcheado y compilado**,
- el egg clik descomprimido + `libclik.so`,
- los datos Planck `.clik`,
- `env_remote.sh` con rutas relativas.

> Licencia: los datos `.clik` de Planck están bajo licencia. Verifica que
> tienes derecho a copiarlos a la máquina destino.

## Paso 2 — Copiar a la máquina destino

Tienes **dos formas**:

### 2a) A un servidor con SSH (scp)

```bash
scp dvch_portable_bundle.tar.gz usuario@servidor:~/
```

### 2b) A la nube gratis SIN SSH (subida por URL temporal)

Si el destino es **GitHub Codespaces / Colab** y no tienes SSH, sube el
bundle a un host temporal con un solo comando (usa solo `curl`, sin cuentas):

```bash
bash deploy/upload_bundle.sh
```

Imprime una **URL** (p.ej. `https://0x0.st/xxxx.tar.gz`). Cópiala; la usarás
en el Paso 3 como `DVCH_BUNDLE_URL`.

## Paso 3 — Instalar y arrancar allí

### 3a) Servidor con SSH

```bash
ssh usuario@servidor
tar -xzf dvch_portable_bundle.tar.gz
cd dvch_bundle_stage
bash repo/deploy/bootstrap_remote.sh
```

### 3b) GitHub Codespaces / Colab (con la URL del Paso 2b) — TODO EN UNO

En el Codespace (o notebook de Colab), con el repo ya clonado:

```bash
export DVCH_BUNDLE_URL='https://0x0.st/xxxx.tar.gz'   # URL del Paso 2b
bash deploy/setup_cloud.sh
```

`setup_cloud.sh` detecta el motor solo y elige la mejor ruta:
1. **Docker** (`python:3.14-slim`) → reproducción exacta, **sin recompilar**;
2. **Python 3.14 local** → usa los `.so` del bundle;
3. sin ninguno de los dos → intenta **recompilar** (más lento).

> **Codespaces ya trae Docker** (via `devcontainer.json`), así que coge la
> Ruta 1 automáticamente y arranca sin tocar nada más.

El bootstrap instala gfortran/OpenMPI/Python, crea un venv, instala Cobaya,
instala el CAMB parcheado incluido, corre el smoke test `DVCH OK` y lanza
la MCMC **full sin recortes**.

## Opciones de nube (lo mas rapido posible)

**Aviso honesto:** "gratis" y "lo mas rapido" chocan para este calculo. Las
capas gratuitas son de 2 vCPU, mas lentas que un PC normal. Lo mas rapido
realista:

| Opcion | Coste real | Velocidad | Compatible con el bundle? |
|---|---|---|---|
| **GCP Free Trial** ($300, 90 dias) | **$0** (gasta ~$1-2 del credito) | **Altisima** (32-64 vCPU) | **Si** (x86_64 + Py3.14 via Docker) |
| **GitHub Codespaces** (120 core-h/mes) | **$0** | Media-alta (4-8 vCPU x86_64) | **Si** (x86_64) — ojo limite de horas |
| GCP e2-micro free tier | $0 | **Baja** (2 vCPU, 1GB RAM) | No: RAM insuficiente (<8GB) |
| Oracle Always Free (Ampere A1) | $0 | Alta (hasta 4 cores/24GB) | **No directo**: es **ARM**, los `.so` son x86_64 -> hay que recompilar |

**Ruta recomendada (gratis y rapida): Google Cloud Free Trial ($300).**
El reparto óptimo es `cadenas = nproc/4` con ~4 hilos OPM por cadena (CAMB
satura a 4 hilos). Tiempos estimados para llegar a R−1<0.03 (modelo
R−1 = C(m)·τ/n_tail con τ≈45 y C(m) calibrado del floor test sintético):

| VM | cadenas | muestras totales | tiempo | crédito GCP usado |
|---|---|---|---|---|
| `c3-standard-16` | 4 | ~158k | ~11 días | ~$25 |
| `c3-standard-32` | 8 | ~175k | ~6 días | ~$30 |
| `c3-standard-64` | 16 | ~230k | ~4 días | ~$55 |
| `c3-standard-96` | 24 | ~302k | ~3.6 días | ~$70 |

Una `c3-standard-64` (o 96) es el punto dulce gratis/rapido.

```bash
# En la VM GCP (Ubuntu 22.04+, x86_64), con Docker instalado:
tar -xzf dvch_portable_bundle.tar.gz
bash deploy/docker_run.sh        # cadenas = nproc/4 automatico
```

> Nota ARM (Oracle Ampere, etc.): si eliges una VM ARM, los `.so` del bundle
> NO cargan. Hay que recompilar clik y CAMB alli (bootstrap + README_INSTALL
> secciones 3-4). Eso anula la ventaja de "rapido".

## Requisito critico: Python 3.14

Las librerias compiladas del bundle (`clik/lkl.cpython-314-*.so`,
`camblib*.so`) son **binarios ABI-especificos de CPython 3.14**. El destino
**debe** tener Python 3.14; con 3.11/3.12/3.13 el `import clik` falla.

Instalacion de Python 3.14 en Ubuntu si no viene por defecto:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.14 python3.14-venv python3.14-dev
# asegura que `python3` sea 3.14, o usa python3.14 -m venv en el bootstrap
```

> Si el destino tiene otra version de Python y no puedes cambiarla, hay que
> **recompilar** clik y CAMB en esa maquina en vez de usar los `.so` del bundle.

## Opciones de hardware

| Opción | Notas |
|---|---|
| Servidor propio / HPC institucional | Lo más simple: ya tiene gfortran/MPI |
| VM de nube grande (16–32 vCPU) | `export DVCH_NCHAINS=8` si hay más cores |
| Cualquier Linux x86_64 | El bundle ya trae los `.so` compilados |

## Monitoreo y convergencia

```bash
tail -f dvch_prod_run.log
# cuando terminen las cadenas:
python3 _r1_diagnostic.py dvch_prod     # Cobaya-R1, classic GR, ESS
python3 _proposal_check.py              # verifica que la propuesta sigue siendo optima
```

Criterio de la fase 3: **Cobaya R−1 < 0.03** (equivalente a ESS ≈ 1400/param).
R−1<0.01 costaría ~33 días de cómputo y no aporta precisión científica
observable; no se persigue. No se declara convergencia hasta tener las cadenas
reales.
EOF