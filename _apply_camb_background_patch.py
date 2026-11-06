#!/usr/bin/env python3
"""Aplica el parche DVCH de FONDO al CAMB de WSL.

Modifica (in place, con aserciones de coincidencia):
  fortran/DVCHModel.f90       <- reemplazado por camb_patch/DVCHModel.f90
  fortran/results.f90         <- use DVCHModel + grho_no_de + densidades de salida
  fortran/equations.f90       <- dtauda, derivs, derivsv

Uso (en WSL):  python3 _apply_camb_background_patch.py [CAMB_ROOT]
"""
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/DVCH-external/CAMB-master"
HERE = "/mnt/d/DVCH13-1"


def patch(path, subs):
    with open(path) as fh:
        s = fh.read()
    applied = 0
    for old, new, expected in subs:
        n = s.count(old)
        if n == 0 and s.count(new) >= 1:
            continue  # ya estaba aplicado (idempotente)
        if n != expected:
            sys.exit(f"FAIL {path}: patron encontrado {n} veces (esperado "
                     f"{expected}):\n{old[:200]}")
        s = s.replace(old, new)
        applied += 1
    with open(path, "w") as fh:
        fh.write(s)
    print(f"OK {path}: {applied} ediciones nuevas de {len(subs)}")


DVCH_CALL_STATE = """        call DVCHBackground(a, (State%grhoc + State%grhob)/State%grhocrit, &
            State%grhov/State%grhocrit, &
            (State%grhog + State%grhornomass + &
             sum(State%grhormass(1:State%CP%Nu_mass_eigenstates)))/State%grhocrit, &
            State%CP%DVCH_n, State%CP%DVCH_beta, dvch_x, dvch_u, dvch_e2, dvch_ok)"""

# ---------------- equations.f90 ----------------
patch(f"{ROOT}/fortran/equations.f90", [
    # dtauda
    ("""    real(dl) :: dtauda, grhoa2, grhov_t

    call this%CP%DarkEnergy%BackgroundDensityAndPressure(this%grhov, a, grhov_t)
""",
     """    real(dl) :: dtauda, grhoa2, grhov_t
    real(dl) :: dvch_x, dvch_u, dvch_e2
    logical :: dvch_ok

    if (this%CP%DVCH_flag) then
        ! DVCH: el vacio sigue la cerradura implicita y la materia la
        ! ecuacion de conservacion CDM-vacio (tabla cacheada del modulo).
        call DVCHBackground(a, (this%grhoc + this%grhob)/this%grhocrit, &
            this%grhov/this%grhocrit, &
            (this%grhog + this%grhornomass + &
             sum(this%grhormass(1:this%CP%Nu_mass_eigenstates)))/this%grhocrit, &
            this%CP%DVCH_n, this%CP%DVCH_beta, dvch_x, dvch_u, dvch_e2, dvch_ok)
        if (.not. dvch_ok) then
            call GlobalError('DVCH background: la integracion de la ODE fallo', error_unsupported_params)
            dtauda = 0
            return
        end if
        grhov_t = this%grhov*dvch_u*a*a
    else
        call this%CP%DarkEnergy%BackgroundDensityAndPressure(this%grhov, a, grhov_t)
    end if
""", 1),
    # derivs: declaraciones
    ("""    real(dl) dgq, grhob_t, grhor_t, grhoc_t, grhog_t, grhov_t, grhonu_t, sigma, polter""",
     """    real(dl) dgq, grhob_t, grhor_t, grhoc_t, grhog_t, grhov_t, grhonu_t, sigma, polter
    real(dl) dvch_x, dvch_u, dvch_e2
    logical dvch_ok""", 1),
    # derivs: densidades
    ("""    if (EV%is_cosmological_constant) then
        grhov_t = State%grhov*a2
        w_dark_energy_t = -1_dl
    else
        call State%CP%DarkEnergy%BackgroundDensityAndPressure(State%grhov, a, grhov_t, w_dark_energy_t)
    end if""",
     f"""    if (State%CP%DVCH_flag) then
{DVCH_CALL_STATE}
        if (.not. dvch_ok) error stop 'DVCH background: la integracion de la ODE fallo'
        grhob_t = grhob_t*dvch_x
        grhoc_t = grhoc_t*dvch_x
        grhov_t = State%grhov*a2*dvch_u
        w_dark_energy_t = -1_dl
    else if (EV%is_cosmological_constant) then
        grhov_t = State%grhov*a2
        w_dark_energy_t = -1_dl
    else
        call State%CP%DarkEnergy%BackgroundDensityAndPressure(State%grhov, a, grhov_t, w_dark_energy_t)
    end if""", 1),
    # derivsv: declaraciones
    ("""    real(dl) grhob_t, grhor_t, grhoc_t, grhog_t, grhov_t, polter""",
     """    real(dl) grhob_t, grhor_t, grhoc_t, grhog_t, grhov_t, polter
    real(dl) dvch_x, dvch_u, dvch_e2
    logical dvch_ok""", 1),
    # derivsv: densidades
    ("""    call CP%DarkEnergy%BackgroundDensityAndPressure(State%grhov, a, grhov_t, w_dark_energy_t)

    grho = grhob_t + grhoc_t + grhor_t + grhog_t + grhov_t""",
     f"""    if (CP%DVCH_flag) then
{DVCH_CALL_STATE.replace('State%CP%', 'CP%')}
        if (.not. dvch_ok) error stop 'DVCH background: la integracion de la ODE fallo'
        grhob_t = grhob_t*dvch_x
        grhoc_t = grhoc_t*dvch_x
        grhov_t = State%grhov*a2*dvch_u
        w_dark_energy_t = -1_dl
    else
        call CP%DarkEnergy%BackgroundDensityAndPressure(State%grhov, a, grhov_t, w_dark_energy_t)
    end if

    grho = grhob_t + grhoc_t + grhor_t + grhog_t + grhov_t""", 1),
])

# ---------------- results.f90 ----------------
patch(f"{ROOT}/fortran/results.f90", [
    ("    module results\n", "    module results\n    use DVCHModel\n", 1),
    # grho_no_de
    ("""    grhoa2 = this%grhok*a**2 + (this%grhoc + this%grhob)*a + this%grhog + this%grhornomass""",
     """    if (this%CP%DVCH_flag) then
        ! DVCH: la materia evoluciona con la conservacion CDM-vacio.
        grhoa2 = this%grhok*a**2 + (this%grhoc + this%grhob)*a* &
            DVCHMatterFraction(a, (this%grhoc + this%grhob)/this%grhocrit, &
            this%grhov/this%grhocrit, &
            (this%grhog + this%grhornomass + &
             sum(this%grhormass(1:this%CP%Nu_mass_eigenstates)))/this%grhocrit, &
            this%CP%DVCH_n, this%CP%DVCH_beta) + this%grhog + this%grhornomass
    else
        grhoa2 = this%grhok*a**2 + (this%grhoc + this%grhob)*a + this%grhog + this%grhornomass
    end if""", 1),
    # GetBackgroundDensities: llamada DE + filas de materia
    ("""    real(dl) :: grhov_t, rhonu, grhonu, a
    real(dl), intent(out) :: densities(8, n)
    integer nu_i, i""",
     """    real(dl) :: grhov_t, rhonu, grhonu, a
    real(dl) :: dvch_x, dvch_u, dvch_e2
    real(dl), intent(out) :: densities(8, n)
    integer nu_i, i""", 1),
    ("""        call this%CP%DarkEnergy%BackgroundDensityAndPressure(this%grhov, a, grhov_t)""",
     """        if (this%CP%DVCH_flag) then
            call DVCHBackground(a, (this%grhoc + this%grhob)/this%grhocrit, &
               this%grhov/this%grhocrit, &
               (this%grhog + this%grhornomass + &
                sum(this%grhormass(1:this%CP%Nu_mass_eigenstates)))/this%grhocrit, &
               this%CP%DVCH_n, this%CP%DVCH_beta, dvch_x, dvch_u, dvch_e2)
            ! convencion a^2 (la fila 8 multiplica despues por a^2)
            grhov_t = this%grhov*dvch_u*a*a
        else
            call this%CP%DarkEnergy%BackgroundDensityAndPressure(this%grhov, a, grhov_t)
        end if""", 1),
    ("""        densities(3, i) = this%grhoc*a
        densities(4, i) = this%grhob*a""",
     """        if (this%CP%DVCH_flag) then
            densities(3, i) = this%grhoc*a*dvch_x
            densities(4, i) = this%grhob*a*dvch_x
        else
            densities(3, i) = this%grhoc*a
            densities(4, i) = this%grhob*a
        end if""", 1),
    # DarkEnergyStressEnergy (salida: rho_de(a)/rho0 y w)
    ("""    do i = 1, n
        call this%CP%DarkEnergy%BackgroundDensityAndPressure(1._dl, a(i), grhov_t(i), w(i))
    end do""",
     """    do i = 1, n
        if (this%CP%DVCH_flag) then
            call DVCHBackground(a(i), (this%grhoc + this%grhob)/this%grhocrit, &
                this%grhov/this%grhocrit, &
                (this%grhog + this%grhornomass + &
                 sum(this%grhormass(1:this%CP%Nu_mass_eigenstates)))/this%grhocrit, &
                this%CP%DVCH_n, this%CP%DVCH_beta, dvch_x, dvch_u, dvch_e2)
            grhov_t(i) = dvch_u
            w(i) = -1._dl
        else
            call this%CP%DarkEnergy%BackgroundDensityAndPressure(1._dl, a(i), grhov_t(i), w(i))
        end if
    end do""", 1),
])

# ---------------- DVCHModel.f90 ----------------
shutil.copyfile(f"{HERE}/camb_patch/DVCHModel.f90", f"{ROOT}/fortran/DVCHModel.f90")
print("OK DVCHModel.f90 copiado")

print("\nPatch aplicado. Recompila CAMB (python setup.py build_ext --inplace).")
