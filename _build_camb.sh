#!/bin/bash
export FORUTILSPATH=/mnt/d/DVCH-external/forutils
cd /mnt/d/DVCH-external/CAMB-master/fortran || exit 9
make python PYCAMB_OUTPUT_DIR=../camb/ > /tmp/make.log 2>&1
echo "make exit=$?"
grep -iE " error |Error:|warning" /tmp/make.log | head -25
tail -20 /tmp/make.log
ls -la ../camb/camblib*.so
