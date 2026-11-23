#!/bin/bash
cd /mnt/d/DVCH-external/CAMB-master/fortran || exit 9
python3 - <<'PYEOF'
s = open('DVCHModel.f90').read()

old_vars = """    real(dl), save :: om_m0_s, om_l0_s, om_r0_s, n_s, beta_s
    real(dl), save, allocatable :: u_tab(:), lnx_tab(:)"""
new_vars = """    real(dl), save :: om_m0_s, om_l0_s, om_r0_s, n_s, beta_s
    ! Switch de desacoplamiento temprano (variante): S(u) = 1/(1+(z/zc)^p)
    ! multiplica solo la DESVIACION del vacio respecto de LCDM.
    ! S=1 (por defecto, z_c enorme) recupera la cerradura original.
    real(dl), save :: zc_s = 1.e6_dl, p_s = 10._dl
    logical, save :: switch_read = .false.
    real(dl), save, allocatable :: u_tab(:), lnx_tab(:)"""
assert s.count(old_vars) == 1, 'vars'
s = s.replace(old_vars, new_vars)

old_read = """contains

    !----------------------------------------------------------
    ! Cerradura algebraica"""
new_read = """contains

    !----------------------------------------------------------
    ! Lee (z_c, p) del entorno (DVCH_ZC, DVCH_P). Por defecto z_c enorme
    ! => S = 1 => cerradura original sin cambio.
    !----------------------------------------------------------
    subroutine read_switch()
        character(len=32) :: val
        if (switch_read) return
        call get_environment_variable('DVCH_ZC', val)
        if (len_trim(val) > 0) read(val, *) zc_s
        call get_environment_variable('DVCH_P', val)
        if (len_trim(val) > 0) read(val, *) p_s
        switch_read = .true.
        write(*, *) '[DVCH] switch desacoplamiento: z_c = ', zc_s, ' p = ', p_s
    end subroutine read_switch

    function switch_S(u) result(S)
        real(dl), intent(in) :: u
        real(dl) :: S, uc
        uc = log(1._dl + zc_s)
        S = 1._dl / (1._dl + exp(p_s * (u - uc)))
        if (S > 1._dl) S = 1._dl
        if (S < 0._dl) S = 0._dl
    end function switch_S

    !----------------------------------------------------------
    ! Cerradura algebraica"""
assert s.count(old_read) == 1, 'read'
s = s.replace(old_read, new_read)

old_decl = """        real(dl), intent(out) :: e2, vac
        real(dl) :: rad, base, A, B, bq, disc"""
new_decl = """        real(dl), intent(out) :: e2, vac
        real(dl) :: rad, base, A, B, bq, disc, S_v, e2_new, vac_raw, A2
        integer :: it"""
assert s.count(old_decl) == 1, 'decl'
s = s.replace(old_decl, new_decl)

old_cl = """        rad = om_r0_s * exp(4._dl * u)
        B = rad + om_m0_s * x
        base = om_l0_s * x**n_s
        if (beta_s <= 0._dl) then
            e2 = B + base
            vac = base
        else
            A = base * (1._dl + beta_s)
            bq = 1._dl - beta_s * B
            disc = bq * bq + 4._dl * beta_s * (A + B)
            if (disc < 0._dl .or. x <= 0._dl) then
                e2 = B + A
            else
                e2 = (-bq + sqrt(disc)) / (2._dl * beta_s)
            end if
            vac = A / (1._dl + beta_s * e2)
        end if
    end subroutine dvch_closure"""
new_cl = """        call read_switch()
        rad = om_r0_s * exp(4._dl * u)
        B = rad + om_m0_s * x
        base = om_l0_s * x**n_s
        if (beta_s <= 0._dl) then
            e2 = B + base
            vac = base
        else
            A = base * (1._dl + beta_s)
            bq = 1._dl - beta_s * B
            disc = bq * bq + 4._dl * beta_s * (A + B)
            if (disc < 0._dl .or. x <= 0._dl) then
                e2 = B + A
            else
                e2 = (-bq + sqrt(disc)) / (2._dl * beta_s)
            end if
            vac = A / (1._dl + beta_s * e2)
        end if
        ! Variante de desacoplamiento: solo la desviacion del vacio respecto
        ! de LCDM (vac=1) se apaga a alto z; a z << z_c se recupera todo.
        S_v = switch_S(u)
        if (S_v >= 1._dl - 1e-14_dl) return
        vac = 1._dl + S_v * (vac - 1._dl)
        do it = 1, 80
            e2_new = rad + om_m0_s * x + om_l0_s * vac
            if (abs(e2_new - e2) < 1e-13_dl * max(1._dl, e2)) exit
            e2 = e2_new
            base = om_l0_s * x**n_s
            if (beta_s > 0._dl .and. x > 0._dl) then
                A2 = base * (1._dl + beta_s)
                bq = 1._dl - beta_s * (rad + om_m0_s * x)
                disc = bq * bq + 4._dl * beta_s * (A2 + rad + om_m0_s * x)
                if (disc >= 0._dl) then
                    vac_raw = A2 / (1._dl + beta_s * e2)
                else
                    vac_raw = A2
                end if
            else
                vac_raw = base
            end if
            vac = 1._dl + S_v * (vac_raw - 1._dl)
        end do
    end subroutine dvch_closure"""
assert s.count(old_cl) == 1, 'closure'
s = s.replace(old_cl, new_cl)

old_dv = """        dvac_dm = vac * n_s / (om_m0_s * x)
        dvac_de2 = -om_l0_s * x**n_s * (1._dl + beta_s) * beta_s / &
            (1._dl + beta_s * e2)**2"""
new_dv = """        S_v = switch_S(u)
        dvac_dm = vac * n_s / (om_m0_s * x)
        dvac_de2 = -om_l0_s * x**n_s * (1._dl + beta_s) * beta_s / &
            (1._dl + beta_s * e2)**2
        ! con el switch, la desviacion del vacio escala por S, igual que
        ! en la cerradura (consistencia de la ODE)
        dvac_dm = 1._dl + S_v * (dvac_dm - 1._dl)
        dvac_de2 = S_v * dvac_de2"""
assert s.count(old_dv) == 1, 'dv'
s = s.replace(old_dv, new_dv)

s = s.replace("""        real(dl) :: derv, e2, vac
        real(dl) :: dvac_dm, dvac_de2, denom, coeff, explicit""",
"""        real(dl) :: derv, e2, vac
        real(dl) :: dvac_dm, dvac_de2, denom, coeff, explicit, S_v""")

open('DVCHModel.f90', 'w').write(s)
print('switch anadido OK')
PYEOF
echo "patch_exit=$?"
gfortran -cpp -ffree-line-length-none -fmax-errors=8 -JReleaselib -IReleaselib -c DVCHModel.f90 -o /tmp/dvchswitch.o 2>&1 | head -6
echo compile_check_done