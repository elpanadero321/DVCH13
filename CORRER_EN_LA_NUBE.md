# CORRER_EN_LA_NUBE.md - Reproducir DVCH gratis en la nube

> Guia practica para correr la **bateria completa de DVCH** y el **MCMC de Planck**
> sin PC potente, usando plataformas gratuitas. Complementa a `RUN_FULL_BATTERY.md`
> (referencia autoritativa del pipeline) y al script turnkey `correr_todo.sh`.
>
> **Politica de integridad (del repo).** Nunca se reportan priors, espectros,
> posteriors ni numeros de convergencia que no provengan de una corrida real y
> convergida. Esta guia solo explica *donde* y *como* correr; no afirma ningun
> resultado cientifico.

---

## 0. Un comando para todo: `correr_todo.sh`

En la raiz del repo, con la carpeta `pkgs/` al lado:

```bash
bash correr_todo.sh
```

El script instala dependencias, verifica (preflight + clik smoke), corre la
bateria, corre el MCMC de Planck y comprueba convergencia. Se detiene ante el
primer error y deja todo en `correr_todo.log`.

Variables opcionales:

- `DVCH_SKIP_INSTALL=1`  salta la instalacion (si ya instalaste antes).
- `DVCH_SKIP_CMB=1`      corre **solo la parte de fondo** (CC + BAO), sin Planck.
  Util en plataformas sin `pkgs/DataRelease-main.zip`.

---

## 1. Requisitos que pide el pipeline completo

- Linux, **>= 4 nucleos** y **>= 16 GB RAM** (la guia recomienda >= 8 nucleos).
- Internet (o wheels pre-descargados), `gfortran` y MPI (`mpirun`).
- La carpeta `pkgs/` con los archivos de dependencias, incluido
  **`DataRelease-main.zip`** (~509 MB, datos Planck high-l). **Sin este archivo la
  parte CMB no puede correr en ninguna plataforma.**

---

## 2. Opciones gratuitas (de mas a menos recomendada)

### 2.1 Oracle Cloud - "Always Free"  (recomendada para el run completo)

- VM Linux gratis de forma permanente (ARM Ampere A1, hasta ~4 nucleos / 24 GB RAM),
  con acceso root para instalar `gfortran`, MPI y todo lo necesario.
- Es la unica gratuita que permite el **MCMC largo** sin corte por tiempo, porque la
  VM es tuya y persiste.
- Contra: arquitectura ARM (no Intel); algun build (CAMB/clik) puede requerir un
  ajuste menor. Pide registrar tarjeta (no se cobra en el tier gratis).

Pasos resumidos:

```bash
# En la VM (Ubuntu/Oracle Linux):
sudo apt update && sudo apt install -y git build-essential gfortran openmpi-bin libopenmpi-dev python3-venv unzip
git clone https://github.com/elpanadero321/DVCH13.git
cd DVCH13
# sube aqui la carpeta pkgs/ (scp o rclone), luego:
bash correr_todo.sh
```

### 2.2 Kaggle Notebooks  (la mas facil; ideal para la parte de fondo)

- Gratis, sin tarjeta. Da **4 nucleos CPU + 30 GB RAM**, pero **corta a ~9 h por
  sesion** (30 h/semana).
- Perfecta para la parte que se verifica bien (fondo + CC + BAO). El MCMC largo de
  Planck no cabe comodo por el limite de tiempo y MPI es complicado ahi.

En una celda del notebook:

```bash
!git clone https://github.com/elpanadero321/DVCH13.git
%cd DVCH13
# sube pkgs/ como "Dataset" de Kaggle y enlazala, luego:
!DVCH_SKIP_CMB=1 bash correr_todo.sh
```

### 2.3 Google Cloud - trial de $300  (la mas potente, temporal)

- $300 de credito por 90 dias. Permite crear una VM Linux de **8+ nucleos**, justo
  lo que pide la guia. Contra: no es permanente y pide tarjeta.
- Mismos pasos que Oracle (2.1), pero con hardware x86 y mas nucleos.

### 2.4 Google Colab  (no recomendado para esto)

- Solo ~2 nucleos, se desconecta seguido y no aguanta el run largo. Sirve como mucho
  para pruebas rapidas de la parte de fondo con `DVCH_SKIP_CMB=1`.

---

## 3. Al terminar

Reune los `.csv`, cadenas (`chains`) y logs generados y compartelos. Con esos
numeros **reales y convergidos** se actualizan las tablas y figuras del manuscrito.
No se reportan resultados de una corrida no convergida.
