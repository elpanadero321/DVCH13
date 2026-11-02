import re
import subprocess

tex = open('/mnt/d/DVCH13-1/DVCH.tex').read()
names = set()
for m in re.findall(r'texttt\{([a-zA-Z0-9\\_.]+)\}', tex):
    name = m.replace('\\_', '_')
    names.add(name)
with_ext = [n for n in names if any(n.endswith(e) for e in
            ('.py', '.sh', '.csv', '.json', '.f90', '.yaml', '.npy', '.ipynb', '.md'))]

# lista explicita para evitar falsos positivos: solo archivos citados con extension
bad = []
for name in sorted(with_ext):
    r = subprocess.run(['bash', '-lc',
                        f'find /mnt/d/DVCH13-1 -maxdepth 3 -name "{name}" '
                        f'-not -path "*/_archive/*" 2>/dev/null | head -1'],
                       capture_output=True, text=True)
    if not r.stdout.strip():
        bad.append(name)

print("archivos citados:", len(with_ext))
print()
if bad:
    print("CITADOS PERO INEXISTENTES:")
    for n in bad:
        print(f"  {n}")
else:
    print("OK: todos los archivos citados existen en el arbol activo")
