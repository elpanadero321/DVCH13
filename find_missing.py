import subprocess
names = ['DVCH_easy_colab.ipynb', 'dvch_class_integration_notes.md',
         'dvch_cobaya.yaml', 'dvch_execution_validation_summary.csv',
         'dvch_theory.py']
for n in names:
    r = subprocess.run(['bash', '-lc',
                        f'find /mnt/d/DVCH13-1 -name "{n}" 2>/dev/null'],
                       capture_output=True, text=True)
    print(f"{n}:")
    found = r.stdout.strip().split('\n') if r.stdout.strip() else []
    for f in found:
        print(f"    {f}")
    if not found:
        print("    NO EXISTE EN NINGUN LADO")
    print()
