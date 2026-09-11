# ============================================================
# Dockerfile — DVCH MCMC full (4 cadenas x 2000, Planck 2018)
# Para correr en CUALQUIER nube con un solo comando.
#
# Requiere el bundle ya descomprimido junto a este Dockerfile,
# o montarlo como volumen. Ver deploy/docker_run.sh
#
# CRITICO: el bundle trae .so ABI-especificas de CPython 3.14.
# La imagen base DEBE ser x86_64 + Python 3.14.
# ============================================================
FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=1

# gfortran runtime (libgfortran5) para CAMB y clik; OpenMPI para las cadenas.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gfortran libgfortran5 libopenmpi-dev openmpi-bin \
        liblapack3 libfftw3-double3 libfftw3-single3 \
        procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# El bundle se monta/copia aqui (contiene CAMB-master, clik, planck-data, repo)
WORKDIR /opt/dvch
COPY dvch_bundle_stage/ /opt/dvch/

# deps Python. OJO: NO instalamos camb desde pip; usamos el CAMB del bundle.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy scipy pandas matplotlib getdist mpi4py pyyaml cobaya==3.6.2

# Verifica que CAMB parcheado y clik cargan ANTES de gastar horas de cadena.
RUN . /opt/dvch/env_remote.sh && python3 -c "\
import camb; p=camb.CAMBparams(); \
p.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.12); \
p.DVCH_flag=True; p.DVCH_n=0.2; p.DVCH_beta=1e-4; \
print('DVCH OK')" && python3 -c "import clik; print('CLIK OK')"

# Arranque: full run, sin recortes.
COPY deploy/entrypoint.sh /opt/dvch/entrypoint.sh
RUN chmod +x /opt/dvch/entrypoint.sh
ENTRYPOINT ["/opt/dvch/entrypoint.sh"]