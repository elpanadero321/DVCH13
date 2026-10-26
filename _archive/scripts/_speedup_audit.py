#!/usr/bin/env python3
"""¿Hay margen real de speedup en esta maquina?

Analiza las palancas que quedan:
 1. layout de hilos (ya probado empiricamente: 4x12 default gana)
 2. RAM disponible (limite duro de cadenas)
 3. throughput actual vs teorico de CAMB
"""
import os
import subprocess

# RAM
with open("/proc/meminfo") as f:
    mem = {l.split(":")[0]: int(l.split(":")[1].split()[0])
           for l in f if ":" in l}
free_gb = mem["MemAvailable"] / 1e6
print("RAM disponible: %.1f GB" % free_gb)

# procesos python del MCMC y su RSS
out = subprocess.run(["ps", "-o", "rss,args", "-C", "python3"],
                     capture_output=True, text=True).stdout
rss = [int(l.split()[0]) for l in out.splitlines()
       if "run_dvch_cobaya" in l]
print("cadenas: %d  RSS/cadena: %.0f MB  total: %.1f GB"
      % (len(rss), sum(rss) / max(len(rss), 1) / 1024,
         sum(rss) / 1024 / 1024))
print("cadenas max por RAM (~7.9GB total): %d"
      % int(free_gb * 1024 / (sum(rss) / max(len(rss), 1)) + len(rss)))

# cores
print("cores: %s" % open("/proc/cpuinfo").read().count("processor"))
