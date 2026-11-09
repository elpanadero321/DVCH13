#!/usr/bin/env python3
"""Reescribe la cerradura con switch de forma ANALITICA (cuadratica).

con vac = 1 + S*(A/(1+b*e2) - 1)  y  e2 = rad + Om*x + OL*vac:
    e2 = C + K/(1 + b*e2),   C = rad + Om*x + OL*(1-S),  K = S*OL*A
    => b*e2^2 + (1 - b*C)*e2 - (C + K) = 0   (raiz positiva, sin iterar)

S=1 recupera exactamente la cerradura original (C = rad+Om*x, K = OL*A).
Mucho mas rapida que la version de punto fijo (que hacia la cerradura
4x mas lenta y el chequeo de punto fijo tardaba >45 min).
"""
PATH = "/mnt/d/DVCH-external/CAMB-master/fortran/DVCHModel.f90"
s = open(PATH).read()

old = """            vac = A / (1._dl + beta_s * e2)
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

new = """            vac = A / (1._dl + beta_s * e2)
        end if
        ! Variante de desacoplamiento: vac = 1 + S*(vac - 1), con
        ! e2 = rad + Om*x + OL*vac. Sustituyendo queda una cuadratica
        ! exacta: e2 = C + K/(1+b*e2),  C = rad+Om*x+OL*(1-S),
        ! K = S*OL*A  (A = OL*x^n*(1+b) si b>0, OL*x^n si b=0).
        S_v = switch_S(u)
        if (S_v < 1._dl - 1e-14_dl) then
            if (beta_s > 0._dl) then
                A2 = base * (1._dl + beta_s)
            else
                A2 = base
            end if
            Ckk = rad + om_m0_s * x + om_l0_s * (1._dl - S_v)
            Kk = S_v * om_l0_s * A2
            if (beta_s > 0._dl) then
                bq = 1._dl - beta_s * Ckk
                disc = bq * bq + 4._dl * beta_s * (Ckk + Kk)
                if (disc >= 0._dl) then
                    e2 = (-bq + sqrt(disc)) / (2._dl * beta_s)
                else
                    e2 = Ckk + Kk
                end if
            else
                e2 = Ckk + Kk
            end if
            if (beta_s > 0._dl) then
                vac = 1._dl + S_v * (A2 / (1._dl + beta_s * e2) - 1._dl)
            else
                vac = 1._dl + S_v * (base - 1._dl)
            end if
        end if
    end subroutine dvch_closure"""

assert s.count(old) == 1, "tail no encontrado (¿version iterativa?)"
s = s.replace(old, new)

old = """        real(dl) :: rad, base, A, B, bq, disc, S_v, e2_new, vac_raw, A2
        integer :: it"""
new = """        real(dl) :: rad, base, A, B, bq, disc, S_v, A2, Ckk, Kk"""
assert s.count(old) == 1, "decl"
s = s.replace(old, new)

open(PATH, "w").write(s)
print("cerradura con switch analitica OK")
