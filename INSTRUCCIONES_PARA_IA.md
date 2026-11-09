# Modelo α — instrucciones para la siguiente IA

Sección 3 (otros errores: qué hacer y por qué) y sección 4 (pendientes, paso
a paso) tal como las entregó la IA que corrigió el test de conservación.
Verifique los claims al final del archivo antes de continuar.

---

## 3. Otros errores: qué hacer y por qué

### E1. El Δχ² compara contra un ΛCDM de otra ejecución

`chi2_lcdm` sale de `dvch_late_time_fit_summary.csv`, que usó 4 puntos de
partida, `maxfev=6000` y `xatol=1e-5`. El modelo α usa 10 puntos y 8000
evaluaciones. Un mínimo ΛCDM peor convergido infla el Δχ².

**Arreglo**: en `alpha_model_fit.py`, minimizar `total_chi2_alpha` con α = 0
fijo (4 parámetros libres, mismos puntos de partida). Debe dar
`1416.51 ± 0.01`. Usar ese valor y guardarlo en el CSV de salida.

### E2. El valor de w0wa está escrito a mano y su log está desactualizado

La línea 219 tiene `chi2_lcdm - 3.35` fijo, y `chi2_w0wa.log` es antiguo
(1701 SNe, χ² = 1832).

**Arreglo**:

- Verificar que `w0wa_control.py` usa `dvr.load_pantheon()` (1580 SNe) y
  `dvr.desi_data`.
- Volver a ejecutar `chi2_w0wa.py` y escribir `dvch_w0wa_fit.csv`.
- Leer ese número desde el CSV en `alpha_model_fit.py`.

### E3. La rejilla del diagnóstico de fondo es demasiado gruesa

En `main()`:

- Usar `a = np.linspace(1e-3, 1, 20001)` para el máximo de ρc/(Oc a⁻³).
- Cambiar el `0.844` fijo por `(1 − fb)`.
- Calcular `dH/H` también contra el mejor ajuste ΛCDM (Om = 0.3108,
  H0 = 68.46), no solo contra el mismo Om.

### E4. El MCMC no tiene diagnóstico de convergencia

- Añadir `sampler.get_autocorr_time(tol=0)` y exigir `n_steps > 50·τ_max`.
- Reportar la fracción de aceptación.
- Arrancar desde `bf.x` en lugar del `p0` escrito a mano.
- Guardar con `os.path.join(_HERE, ...)`: ahora mismo `np.save` escribe en el
  directorio desde donde se ejecuta.

### E5. El parche CAMB antiguo acopla también los bariones

`_apply_camb_background_patch.py` (líneas 86 y 108) multiplica `grhob_t` por
`dvch_x`. En el modelo α solo el CDM cambia, así que no hay que copiar esa
línea.

### E6. Notación en los documentos

- "ρ_m + 3Hρ_m = Q" debe ser `dρ_c/dt + 3Hρ_c = +Q` (solo CDM) y
  `dρ_Λ/dt = −Q`.
- Corregir "θ* a salvo" y "CDM +10 %" en `PENDIENTES_ALPHA.md` y
  `SESSION_LOG.txt` con los números de la sección 2 (θ* sube +0.78% en
  α = 0.597; la desviación del CDM es del 13% en z = 0.25).

---

## 4. Pendientes 1–5, paso a paso (en este orden)

### Paso 0 (barato, sin parchear CAMB): el veredicto de fondo

Como el fondo es exactamente una DE desacoplada, se puede usar CAMB estándar:

```python
a = np.geomspace(1e-5, 1, 2000)
w = -1 + 2*alpha*a*(1-a) / (3*(OL + alpha*(1-a)**2))
pars.set_dark_energy_w_a(a, w, dark_energy_model='ppf')
# omch2 = coeficiente de a^-3
```

Los valores de θ*, r_drag y D_M salen exactos.

Ajustar Planck + DESI + Pantheon+ con α libre y comparar con ΛCDM en la misma
configuración.

Si aquí ya Δχ² > +5, el modelo cae sin necesidad de perturbaciones. Las
perturbaciones de este fluido no son las del modelo; solo sirve para el fondo.

### Paso 1: cierre de perturbaciones (CDM geodésico)

Elección: Q^μ = Q·u_c^μ. El CDM sigue siendo geodésico, θ_c = 0, y el gauge
síncrono de CAMB (definido por el CDM en reposo) sigue siendo válido. En ese
gauge δρ_Λ = 0 y δQ = 0, y no hay transferencia de momento (el vacío no lleva
momento). Es el modelo "geodesic CDM" de Wands et al., estable y sin fuerzas
nuevas sobre los bariones.

Ecuación exacta en tiempo conforme:

```
δ_c' = −h'/2 − ℋ·q(a)·δ_c
q(a) = Q/(Hρ_c) = (2α/3)·a(4−5a) / ρ̃_c(a)
ρ̃_c(a) = Oc a^-3 + (2α/3)·a(1−a)      (en unidades de ρ_crit0)
```

En CAMB (`equations.f90` → derivs):

```fortran
clxcdot = -k*z - adotoa*q*clxc
grhoc_t = grhoc/a * fc(a),   con fc = 1 + (2α/(3Oc))·a⁴(1−a)
grhov_t = grhov*a**2 * ρ̃_L(a)/OL
```

Mismos cambios en `dtauda`, `grho_no_de` y `GetBackgroundDensities`
(`results.f90`). No tocar `grhob_t`. Las perturbaciones de DE desactivadas
(w = −1).

Validación, en orden:

1. Con α = 0, los espectros deben ser idénticos a ΛCDM bit a bit.
2. El fondo debe coincidir con el Paso 0 a `< 1e-5` (θ*, r_drag, D_M).
3. Integración independiente en Python para z < 50:

   ```
   δ_c' = −h'/2 − ℋqδ_c
   δ_b' = −h'/2
   h'' + ℋh' = −3ℋ²(Ω_c(a)δ_c + Ω_b(a)δ_b)
   ```

   Esta última es exacta en gauge síncrono porque δρ_Λ = δP_Λ = 0. El
   crecimiento resultante debe coincidir con CAMB a ~1e-3.

### Paso 2: test CMB completo

Cobaya con el CAMB parcheado: `planck_2018_highl_plik.TTTEEE`, `lowl.TT`,
`lowl.EE`, `lensing`, más `bao.desi_2024_bao_all` y `sn.pantheonplus`.

Parámetros libres: α, H0, ω_b, ω_c, A_s, n_s, τ y los nuisance oficiales.
Minimizador en ambos modelos con los mismos datos (reutilizar
`cmb_minimize_compare.py`).

Criterio: `Δχ²_CMB < +5` en el mínimo conjunto. Reportar también S8.

### Paso 3: mecanismo para α

Hoy Q contiene ρ_crit0, una constante "de hoy", así que no es una ley local.

Lo mínimo: reescribir Q como función de cantidades locales,
`Q = H·F(ρ_c, ρ_Λ)` o un vacío que corre con ρ_Λ(H²), y comprobar si alguna
forma covariante sencilla reproduce `(2α/3)a(4−5a)`.

"Solo CDM": no se deriva, se justifica. Los tests del principio de equivalencia
prohíben acoplar los bariones, y el CDM geodésico es la elección consistente.
Hay que decirlo así en el paper.

### Paso 4: BBN

Trivial y basta documentarlo. A T ≈ 1 MeV (a ≈ 2.4e-10):

- ρ_Λ/ρ_r ≈ (OL+α)·a⁴/Or ≈ 4e-35
- la desviación del CDM va como a⁴
- El N_eff efectivo no cambia.

Un script de 3 líneas con esta cuenta es suficiente.

### Paso 5: generalizar para que el ajuste mejore

Regla general, válida para cualquier desviación Δ(a):

```
δρ_c = −(a/3)·dΔ/da
δρ_L = Δ − δρ_c
Q/(Hρ_crit0) = 3δρ_c + a·dδρ_c/da
```

- `(1−a)^m` con `m < 1` no sirve: δρ_c ∝ (1−a)^(m−1) diverge en a = 1.
- Con `m > 1` se mantiene w0 = −1, así que no se gana a w0wa.

Recomendado (análogo de CPL, 2 parámetros): `Δ = α1(1−a) + α2(1−a)²`

```
δρ_c = (a/3)·[α1 + 2α2(1−a)]
Q/(Hρ_crit0) = a·[(4/3)α1 + (2/3)α2(4−5a)]
w0 = −1 + α1/(3·OL)
```

Cuidado: Om es el coeficiente de a⁻³; la materia hoy es `Om + α1/3`.
El régimen tipo DESI corresponde a α1 > 0 y α2 < −α1.
Comprobar en toda la evolución ρ_c > 0 y ρ_L > 0.
Generalizar `verify()` (con el `q_over_h` nuevo) y añadir el parámetro al
MCMC.

Para que sea interesante tiene que bajar de −3.35 en los datos tardíos y pasar
los Pasos 0–2.

---

## Apéndice: verificación local de los claims (hecha por la IA de este repo)

### Poblado por los runners de cobaya (Paso 0 ejecutado, 2026-11-08)

Resultados `chi2_min` (nuisance oficial de plik libre, `minimize
ignore_prior`; 4 likelihoods: plikTTTEEE + commander TT + simall EE +
smica lensing; CAMB estándar en PPF):

```
solo CMB     : LCDM 2761.50 | alpha 2759.90 (alpha=0.148, H0=66.7)
conjunto     : LCDM 4178.59 | alpha 4180.71 (alpha=-0.070, H0=68.09)
solo tardio  : LCDM 1416.51 | alpha 1415.10 (alpha=0.597)
```

Veredicto de fondo: el modelo alpha **no es excluido por el CMB** (frente al
`+5.9e4` del DVCH (n, beta)), pero **no gana**: en el ajuste conjunto
Planck+DESI+Pantheon+CC el mejor `alpha` es `~0` y
`Delta chi2 = +2.13` (1 param extra, `Delta AIC = +4.1`). No hay régimen
donde el modelo sea a la vez no-trivial y mejor que LCDM. Archivos:
`alpha_lcdm_min_free.bestfit.txt`, `alpha_free_min.bestfit.txt`,
`alpha_joint_lcdm_min.bestfit.txt`, `alpha_joint_free_min.bestfit.txt`,
corredores `run_alpha_cobaya.py` / `run_alpha_joint.py` (yaml
`alpha_cobaya_short.yaml`, `alpha_joint.yaml`).

### Verificaciones puntuales

- **Test de conservación corregido**: se ejecutó; pasa con residuos
  1e-10…1e-17 y el test de mutación (vacio con 2α) falla como debe
  (residuo 0.20). La fórmula α y la Q documentada
  `Q/(Hρ_crit0) = (2α/3)a(4−5a)` quedan confirmadas.
- **θ* no está a salvo (confirmado con CAMB estándar, `_diag_alpha_cmb.py`)**:
  a `wm_h2 = 0.1430` fijo, Δθ*/θ* = +0.39% (α=0.3), **+0.78% (α=0.597)**,
  +1.29% (α=1.0) a H0 = 67.36 fijo. Mecanismo: el horizonte sonoro
  `r_drag = ∫_{z_d}^∞ c_s dz/H` solo implica z > z_drag ≈ 1060, donde el
  modelo es idéntico a ΛCDM (CAMB da `r_drag` idéntico hasta 1e-4); solo se
  mueve `D_A*`, y por eso θ* no se cancela. Planck mide θ* al 0.03% ⇒ +0.78%
  son ~26σ. (En el ajuste conjunto esto se compensa bajando H0 y α.)
- **w_eff**: confirmado `w_eff(z=1) = −0.88` con α=0.597 y `w(a=1) = −1`
  exacto (print del mismo script).
- **Desviación del CDM**: 13.04% en z = 0.25 con el mejor ajuste (máximo en
  a = 0.8, no en la rejilla tosca).
- **E1**: ΛCDM minimizado in-script en idénticas condiciones da 1416.51 (=
  pipeline). **E2**: `w0wa` re-derivado y escrito a `dvch_w0wa_fit.csv`
  (1413.16, w0=−0.878, wa=−0.441). **E3/E4**: rejilla fina, comparación vs
  mejor ajuste LCDM, `autocorr_time` (τ=53, ratio 75), aceptación 0.55.
- **Bisectriz de "H0 que conserva θ*" en `_diag_alpha_cmb.py`** no es fiable
  (asume monotonicidad y se va a las cotas); los Δχ² a H0 fijo sí están
  verificados.
