import subprocess
files = ['DVCH_easy_colab.ipynb', 'dvch_theory.py', 'dvch_cobaya.yaml',
         'dvch_execution_validation_summary.csv', 'dvch_class_integration_notes.md']
for f in files:
    r = subprocess.run(['git', 'log', '--all', '--oneline', '--', f],
                       capture_output=True, text=True, cwd='/mnt/d/DVCH13-1')
    print(f"{f}:")
    print("   commits:", r.stdout.strip().split('\n')[0] if r.stdout.strip() else 'NINGUNO')
    # ver si aparece en algun commit
    r2 = subprocess.run(['git', 'log', '--all', '--diff-filter=A', '--format=%H',
                         '--', f], capture_output=True, text=True, cwd='/mnt/d/DVCH13-1')
    commits = r2.stdout.strip().split('\n') if r2.stdout.strip() else []
    if commits:
        c = commits[-1]  # commit mas antiguo que lo anadio
        r3 = subprocess.run(['git', 'show', f'{c}:{f}'], capture_output=True,
                            text=True, cwd='/mnt/d/DVCH13-1')
        print(f"   recuperable de {c[:8]}, {len(r3.stdout)} bytes")
    print()
