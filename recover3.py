import subprocess
files = ['DVCH_easy_colab.ipynb', 'dvch_class_integration_notes.md',
         'dvch_execution_validation_summary.csv']
for f in files:
    r = subprocess.run(['git', 'log', '--all', '--diff-filter=A', '--format=%H', '--', f],
                       capture_output=True, text=True, cwd='/mnt/d/DVCH13-1')
    commits = [c for c in r.stdout.strip().split('\n') if c]
    print(f"=== {f} ===")
    if not commits:
        print("  nunca estuvo en git (commit --all lo busca en todos los commits)")
    else:
        c = commits[-1]
        r2 = subprocess.run(['git', 'show', f'{c}:{f}'], capture_output=True,
                            text=True, cwd='/mnt/d/DVCH13-1')
        print(f"  anadido en {c[:8]}, contenido: {len(r2.stdout)} chars")
        open(f'/mnt/d/DVCH13-1/{f}', 'w').write(r2.stdout)
        print(f"  RECUPERADO a {f}")
    print()
