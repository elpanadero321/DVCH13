#!/usr/bin/env python3
"""Corrige el switch de desacoplamiento (unidades + derivada ODE).

Errores de la primera version:
  1. La "base" de LCDM del vacio es OL (Omega_Lambda), no 1:
       vac = OL + S*(vac_raw - OL)       [no  1 + S*(vac_raw - 1)]
  2. K = S*A2 (A2 ya contiene OL una vez; antes S*OL*A2).
  3. La derivada de la ODE debe escalar como S*dvac_dm (no 1+S*(dv-1)):
     a S=0 la materia tiene que escalar como a^-3 exactamente.
S=1 sigue dando la cerradura original bit a bit.
"""
PATH = "/mnt/d/DVCH-external/CAMB-master/fortran/DVCHModel.f90"
s = open(PATH).read()

# --- cerradura ---
old = """            if (beta_s > 0._dl) then
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
new = """            if (beta_s > 0._dl) then
                A2 = base * (1._dl + beta_s)
            else
                A2 = base
            end if
            ! base de LCDM = OL (no 1): vac = OL + S*(vac_raw - OL),
            ! e2 = C + K/(1+b*e2) con C = rad+Om*x+OL*(1-S), K = S*A2
            Ckk = rad + om_m0_s * x + om_l0_s * (1._dl - S_v)
            Kk = S_v * A2
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
                vac = om_l0_s + S_v * (A2 / (1._dl + beta_s * e2) - om_l0_s)
            else
                vac = om_l0_s + S_v * (base - om_l0_s)
            end if
        end if
    end subroutine dvch_closure"""
assert s.count(old) == 1, "closure-switch no encontrado"
s = s.replace(old, new)

# --- derivada de la ODE ---
old = """        dvac_dm = 1._dl + S_v * (dvac_dm - 1._dl)
        dvac_de2 = S_v * dvac_de2"""
new = """        dvac_dm = S_v * dvac_dm
        dvac_de2 = S_v * dvac_de2"""
assert s.count(old) == 1, "escalado ODE no encontrado"
s = s.replace(old, new)

open(PATH, "w").write(s)
print("switch corregido (unidades + derivada) OK")
