path = "SESSION_LOG.txt"
text = open(path).read()
bad = "  - _diag_alpha_cmb.py: la bisectriz "
if text.rstrip().endswith(bad.strip()):
    fix = (
        "\n(no es monotona); los dchi2 a H0 fijado si lo son.\n"
        "  - run_alpha_joint.py (modo lcdm): el signature mal puesto quitaba Msn\n"
        "    en vez de alpha -> cobaya 'Requirement alpha not provided'. Corregido.\n"
        "  - D_H en la likelihood joint tenia un factor c de mas (1.1e9 Mpc).\n"
    )
    open(path, "w").write(text.rstrip() + "\n" + fix)
    print("fixed")
else:
    print("unexpected tail:", repr(text[-80:]))
