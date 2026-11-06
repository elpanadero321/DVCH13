#!/bin/bash
cd /mnt/d/DVCH13-1 || exit 9
source env.sh > /tmp/env2.out 2>&1
chmod +x launch_dvch_free_minimize.sh
nohup bash launch_dvch_free_minimize.sh > /tmp/dvchfreemin.log 2>&1 &
echo "launched pid=$!"
sleep 20
tail -5 /tmp/dvchfreemin.log
