# Correr la MCMC DVCH completa en OTRA máquina (sin recortes)

Este directorio contiene todo lo necesario para mover la corrida **full**
(4 cadenas × 2000 muestras, Planck 2018 TTTEEE+lowl+lensing + Pantheon+ + BAO)
a un servidor / HPC / VM de nube, sin tocar los parámetros científicos.

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
Una VM `c3-standard-32` (32 vCPU) termina las 4x2000 en ~1-2 h y consume
menos de $2 del credito de $300 -> en la practica gratis.

```bash
# En la VM GCP (Ubuntu 22.04+, x86_64), con Docker instalado:
tar -xzf dvch_portable_bundle.tar.gz
export DVCH_NCHAINS=32
bash deploy/docker_run.sh
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
python3 dvch_planck_chain_diagnostics.py dvch_prod.[1-4].txt
```

Criterio: **R-hat < 1.01 y ESS > 100**. No se declara convergencia hasta
tener las cadenas reales.
EOF