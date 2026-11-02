import re
import subprocess

BASE = '/mnt/d/DVCH13-1'
tex = open(f'{BASE}/DVCH.tex').read()

# Extraer texttt{...} y normalizar \_ a _
names = set()
for m in re.findall(r'texttt\{([a-zA-Z0-9\\_.]+)\}', tex):
    name = m.replace('\\_', '_')
    names.add(name)

with_ext = [n for n in names if any(n.endswith(e) for e in
            ('.py', '.sh', '.csv', '.json', '.f90', '.yaml', '.npy', '.ipynb', '.md'))]

print(f"archivos citados con texttt: {len(with_ext)}")
missing_anywhere = []
missing_active = []
for name in sorted(with_ext):
    r = subprocess.run(['bash', '-lc',
                        f'find /mnt/d/DVCH13-1 -maxdepth 3 -name "{name}" '
                        f'-not -path "*/_archive/*" 2>/dev/null | head -1'],
                       capture_output=True, text=True)
    status = 'ok' if r.stdout.strip() else 'AUSENTE'
    r2 = subprocess.run(['bash', '-lc',
                         f'find /mnt/d/DVCH13-1 -name "{name}" 2>/dev/null | head -1'],
                        capture_output=True, text=True)
    if status == 'ok':
        print(f"  ok      {name}")
    elif r2.stdout.strip():
        print(f"  ARCHIVADO {name} -> {r2.stdout.strip()}")
        missing_active.append(name)
    else:
        print(f"  PERDIDO {name}")
        missing_anywhere.append(name)

print()
print(f"en _archive (recuperables): {len(missing_active)}")
print(f"perdidos del todo: {len(missing_anywhere)}")
if missing_anywhere:
    print("Lista de perdidos:", missing_anywhere)
