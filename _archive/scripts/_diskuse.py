import os
s = os.statvfs("/")
print("WSL / : total %.1f GB  free %.1f GB" % (s.f_blocks * s.f_bsize / 1e9,
                                                s.f_bfree * s.f_bsize / 1e9))
# grandes directorios dentro de WSL
for d in ("/home/danieproyect", "/tmp", "/var", "/root", "/usr"):
    try:
        total = 0
        for root, dirs, files in os.walk(d):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        print("%-30s %.2f GB" % (d, total / 1e9))
    except OSError:
        pass
