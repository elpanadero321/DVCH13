import re
import os
import subprocess

BASE = '/mnt/d/DVCH13-1'
os.chdir(BASE)
tex = open('DVCH.tex').read()

# extraer todos los \texttt{...} (pueden contener \_ escapado)
cited = set()
for m in re.findall(r'\\texttt\{\\?([A-Za-z0-9_.\\]+)\}', tex):
    name = m.replace('\\_', '_').replace('\\\\', '')
    if any(name.endswith(ext) for ext in ('.py', '.sh', '.csv', '.json', '.f90',
                                          '.yaml', '.npy', '.ipynb', '.md', '.tex')):
        cited.add(name)

print(f"archivos citados en el paper: {len(cited)}")
print()
missing = []
for name in sorted(cited):
    # buscar en el arbol excepto _archive
    r = subprocess.run(['bash', '-lc',
                        f'find /mnt/d/DVCH13-1 -maxdepth 3 -name "{name}" '
                        f'-not -path "*/_archive/*" 2>/dev/null | head -1'],
                       capture_output=True, text=True)
    hit = r.stdout.strip()
    if hit:
        pass
    else:
        # existe aunque sea en _archive?
        r2 = subprocess.run(['bash', '-lc',
                             f'find /mnt/d/DVCH13-1 -name "{name}" | head -1'],
                            capture_output=True, text=True)
        if r2.stdout.strip():
            missing.append((name, 'ARCHIVADO (en _archive)'))
        else:
            missing.append((name, 'INEXISTENTE'))

if missing:
    print("PROBLEMAS:")
    for n, why in missing:
        print(f"  {n}: {why}")
else:
    print("todos los archivos citados existen en el arbol activo")

# ademas: los figures referenciados
figs = sorted(set(re.findall(r'figures/([a-zA-Z0-9_]+)\.png', tex)))
print(f"\nfiguras referenciadas: {len(figs)}")
for f in figs:
    p = f'figures/{f}.png'
    if os.path.exists(p):
        sz = os.path.getsize(p)
        print(f"  ok  {p} ({sz} bytes)")
    else:
        print(f"  FALTA {p}")
