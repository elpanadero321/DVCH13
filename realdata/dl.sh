#!/bin/bash
cd /mnt/d/DVCH13-1/realdata
echo "Descargando DESI DR1 BAO..."
curl -sL --max-time 300 -o desi_dr1_bao.txt 'https://raw.githubusercontent.com/cosmodesi/desibaobkg/main/data/desi_dr1_bao.txt' && cat desi_dr1_bao.txt | head -20
