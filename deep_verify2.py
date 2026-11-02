import re
import os
import subprocess

BASE = '/mnt/d/DVCH13-1'
os.chdir(BASE)
tex = open('DVCH.tex').read()

# Metodo robusto: buscar cada palabra de archivo explicitamente
CANDIDATES = []
for ext in ('.py', '.sh', '.csv', '.json', '.f90', '.yaml', '.npy', '.ipynb', '.md'):
    for m in re.findall(r'[A-Za-z0-9_.\\]+\.' + ext.replace('.', r'\.'), tex):
        name = m.replace('\\_', '_')
        CANDIDATES.append(name)

cited = set(CANDIDATES)
print(f"archivos detectados en el paper: {len(cited)}")

missing_active = []
missing_anywhere = []
for name in sorted(cited):
    r = subprocess.run(['bash', '-lc',
                        f'find /mnt/d/DVCH13-1 -maxdepth 3 -name "{name}" '
                        f'-not -path "*/_archive/*" 2>/dev/null | head -1'],
                       capture_output=True, text=True)
    if not r.stdout.strip():
        r2 = subprocess.run(['bash', '-lc',
                             f'find /mnt/d/DVCH13-1 -name "{name}" 2>/dev/null | head -1'],
                            capture_output=True, text=True)
        if r2.stdout.strip():
            missing_active.append((name, r2.stdout.strip()))
        else:
            missing_anywhere.append(name)

if missing_active:
    print("\ncitados pero SOLO en _archive:")
    for n, loc in missing_active:
        print(f"  {n}  -> {loc}")
if missing_anywhere:
    print("\ncitados y NO EXISTEN en ningun lado:")
    for n in missing_anywhere:
        print(f"  {n}")
if not missing_active and not missing_anywhere:
    print("\nOK: todos los archivos detectados existen en el arbol activo")
