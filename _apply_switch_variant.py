#!/usr/bin/env python3
"""Anade la variante de desacoplamiento temprano al CAMB externo.

S(u) = 1/(1+(z/z_c)^p) multiplica solo la DESVIACION del vacio respecto
de LCDM. Por defecto (sin DVCH_ZC) z_c es enorme => S = 1 => cerradura
original exacta. Con DVCH_ZC=5, DVCH_P=10: a z<2.3 (todos los datos
tardios) el modelo es el original; a z>10 es LCDM puro.
Se aplica sobre el DVCHModel.f90 ya parcheado de fondo (idempotente).
"""
import sys

PATH = "/mnt/d/DVCH-external/CAMB-master/fortran/DVCHModel.f90"
s = open(PATH).read()
if "switch_S" in s:
    print("switch ya presente; no se hace nada")
    sys.exit(0)

# 1) variables del switch
old = """    real(dl), save :: om_m0_s, om_l0_s, om_r0_s, n_s, beta_s
    real(dl), save, allocatable :: u_tab(:), lnx_tab(:)"""
new = """    real(dl), save :: om_m0_s, om_l0_s, om_r0_s, n_s, beta_s
    real(dl), save :: zc_s = 1.e6_dl, p_s = 10._dl
    logical, save :: switch_read = .false.
    real(dl), save, allocatable :: u_tab(:), lnx_tab(:)"""
assert s.count(old) == 1, "vars"
s = s.replace(old, new)

# 2) subrutinas del switch antes de la cerradura
old = """contains

    !----------------------------------------------------------
    ! Cerradura algebraica"""
new = """contains

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
assert s.count(old) == 1, "read"
s = s.replace(old, new)

# 3) declaraciones en la cerradura
old = """        real(dl), intent(out) :: e2, vac
        real(dl) :: rad, base, A, B, bq, disc"""
new = """        real(dl), intent(out) :: e2, vac
        real(dl) :: rad, base, A, B, bq, disc, S_v, e2_new, vac_raw, A2
        integer :: it"""
assert s.count(old) == 1, "decl"
s = s.replace(old, new)

# 4) llamada al switch + modificacion de la cerradura
old = """        rad = om_r0_s * exp(4._dl * u)
        B = rad + om_m0_s * x
        base = om_l0_s * x**n_s
        if (beta_s <= 0._dl) then"""
new = """        call read_switch()
        rad = om_r0_s * exp(4._dl * u)
        B = rad + om_m0_s * x
        base = om_l0_s * x**n_s
        if (beta_s <= 0._dl) then"""
assert s.count(old) == 1, "cl-head"
s = s.replace(old, new)

old = """            vac = A / (1._dl + beta_s * e2)
        end if
    end subroutine dvch_closure"""
new = """            vac = A / (1._dl + beta_s * e2)
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
assert s.count(old) == 1, "cl-tail"
s = s.replace(old, new)

# 5) derivadas de la ODE escaladas por S
old = """        dvac_dm = vac * n_s / (om_m0_s * x)
        dvac_de2 = -om_l0_s * x**n_s * (1._dl + beta_s) * beta_s / &
            (1._dl + beta_s * e2)**2"""
new = """        S_v = switch_S(u)
        dvac_dm = vac * n_s / (om_m0_s * x)
        dvac_de2 = -om_l0_s * x**n_s * (1._dl + beta_s) * beta_s / &
            (1._dl + beta_s * e2)**2
        dvac_dm = 1._dl + S_v * (dvac_dm - 1._dl)
        dvac_de2 = S_v * dvac_de2"""
assert s.count(old) == 1, "dv"
s = s.replace(old, new)

old = """        real(dl) :: derv, e2, vac
        real(dl) :: dvac_dm, dvac_de2, denom, coeff, explicit"""
new = """        real(dl) :: derv, e2, vac
        real(dl) :: dvac_dm, dvac_de2, denom, coeff, explicit, S_v"""
assert s.count(old) == 1, "dxdu-decl"
s = s.replace(old, new)

open(PATH, "w").write(s)
print("switch anadido OK a", PATH)
