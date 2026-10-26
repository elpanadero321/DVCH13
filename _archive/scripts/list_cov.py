#!/bin/bash
cd /mnt/d/DVCH13-1/realdata
curl -sL --max-time 90 'https://api.github.com/repos/PantheonPlusSH0ES/DataRelease/contents/Pantheon%2B_Data/4_DISTANCES_AND_COVAR' |
  grep -a -o '"name": "[^"]*"' | head -20
