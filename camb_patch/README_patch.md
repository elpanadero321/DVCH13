# Parche DVCH para CAMB

Este parche añade la física DVCH (interacción CDM-vacío con índice de tracking `n` y supresión de curvatura `beta`) al CAMB estándar. Consiste en **5 cambios** sobre un clon limpio de CAMB.

## Archivos del parche

- `DVCHModel.f90` — módulo Fortran **nuevo** (copiar a `fortran/`).

## Cambios manuales (4 archivos)

### 1. `fortran/equations.f90` — añadir `use DVCHModel` (2 lugares)

**Lugar A** — en la función `dtauda` (al inicio del archivo), después de `use DarkEnergyInterface`:

```fortran
    function dtauda(this, a)
    use results
    use DarkEnergyInterface
    use DVCHModel          ! <-- AÑADIR ESTA LÍNEA
    implicit none
```

**Lugar B** — en el módulo `GaugeInterface`, después de `use MassiveNu`:

```fortran
    module GaugeInterface
    use precision
    use results
    use MassiveNu
    use DVCHModel          ! <-- AÑADIR ESTA LÍNEA
    use DarkEnergyInterface
```

### 2. `fortran/equations.f90` — añadir el bloque DVCH en la ecuación CDM

Busca la sección `! CDM equation of motion` (alrededor de la línea 2445). El código original es:

```fortran
    ! CDM equation of motion
    clxcdot = -k*z
    ayprime(ix_clxc) = clxcdot
```

Reemplázalo por:

```fortran
    ! CDM equation of motion
    clxcdot = -k*z
    if (State%CP%DVCH_flag) then
        ! Use the instantaneous CAMB background fractions. The vacuum
        ! perturbation is zero in the adopted vacuum rest frame.
        dvch_E = sqrt(grho/(State%grhocrit*a2))
        dvch_om_m = grhoc_t/grho
        dvch_om_l = grhov_t/grho
        dvch_om_r = (grhog_t + grhor_t)/grho
        dvch_delta_E = 0._dl
        call DVCHInteraction(dvch_E, dvch_om_m, dvch_om_l, dvch_om_r, clxc, &
            dvch_delta_E, State%CP%DVCH_n, State%CP%DVCH_beta, &
            dvch_qtilde, dvch_delta_qtilde)
        clxcdot = clxcdot + 3._dl*adotoa*dvch_qtilde/(dvch_E*dvch_om_m)*clxc &
            - 3._dl*adotoa*dvch_delta_qtilde/(dvch_E*dvch_om_m)
    end if
    ayprime(ix_clxc) = clxcdot
```

> **Nota:** Las variables locales `dvch_*` deben declararse en la subrutina que contiene la ecuación CDM (la subrutina `derivs`/`dtauda` de perturbaciones). Busca la sección de declaraciones de variables locales (después de `real(dl) dgrho_de, dgq_de, cs2_de`) y añade:
> ```fortran
>     real(dl) dvch_qtilde, dvch_delta_qtilde, dvch_E
>     real(dl) dvch_om_m, dvch_om_l, dvch_om_r, dvch_delta_E
> ```

### 3. `fortran/model.f90` — añadir los 3 parámetros

Busca la sección de parámetros del modelo (alrededor de la línea 145). Añade después de los parámetros de densidad:

```fortran
        logical :: DVCH_flag = .false. ! Enable the DVCH CDM perturbation closure
        real(dl) :: DVCH_n = 0.2_dl
        real(dl) :: DVCH_beta = 1.e-4_dl
```

### 4. `fortran/Makefile_main` — añadir `DVCHModel` a `SOURCEFILES`

Busca la línea `SOURCEFILES = ...` (alrededor de la línea 35). Añade `DVCHModel` a la lista:

```makefile
SOURCEFILES      = constants config classes MathUtils RungeKuttaDP45 DarkAge21cm DVCHModel \
```

### 5. `camb/model.py` — exponer los 3 parámetros a Python

Busca la lista de parámetros (alrededor de la línea 333, después de `("H0", ...)`). Añade:

```python
        ("DVCH_flag", c_bool, "Enable the DVCH CDM perturbation closure"),
        ("DVCH_n", c_double, "DVCH tracking index"),
        ("DVCH_beta", c_double, "DVCH curvature suppression parameter"),
```

---

## Compilar

```bash
cd CAMB
python setup.py build
python setup.py install
# o en modo desarrollo:
pip install -e .
```

## Verificar

```bash
python -c "import camb; p=camb.CAMBparams(); p.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.12); p.DVCH_flag=True; p.DVCH_n=0.2; p.DVCH_beta=1e-4; print('DVCH OK')"
```

Si no hay error, el parche está correctamente aplicado.