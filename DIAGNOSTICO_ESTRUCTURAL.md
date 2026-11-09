# DVCH — Diagnóstico estructural completo

Documento autocontenido: la hipótesis, las fórmulas, el problema, todo lo que
se probó y los números. Para que alguien externo pueda juzgar y, eventualmente,
llegar a una hipótesis/fórmula mejor.

---

## 1. Veredicto (resumen)

La hipótesis **no es reparable con correcciones de datos ni con variantes de
la fórmula que mantengan la idea central**. El mecanismo físico de la
cerradura (vacío acoplado a la materia con conservación de energía) produce
siempre una violación permanente de `rho_m * a^3` (la materia deja de crecer
como `a^-3` en el pasado y el déficit se "fosiliza"). El régimen de parámetros
que mejora el ajuste tardío (Delta chi2 = -3.6) exige un déficit de materia
del 15-30%, que el CMB destruye (`chi2` de +1e4 a +1e5). El régimen donde el
déficit desaparece (`n < 0.1`) es indistinguible de LCDM.

---

## 2. La hipótesis y las fórmulas

### 2.1 Notación

- `Om`, `OL`, `Or`: densidades adimensionales actuales (materia total, vacío,
  radiación), `Om + OL + Or = 1`.
- `m(z) = rho_m(z) / rho_crit(0)` (materia adimensional), `x(z) = m(z)/Om`.
- `U(z) = rho_L(z) / rho_L(0)` (vacío normalizado a hoy).
- `E2(z) = H^2(z)/H0^2 = Or*(1+z)^4 + m(z) + rho_L(z)/rho_crit(0)`.

### 2.2 La cerradura del vacío (la hipótesis del paper)

El vacío depende de la materia a través del cociente `x^n`, normalizado por
`beta*E2`:

```
rho_L(z)/rho_crit0 = OL * x(z)^n * (1+beta) / (1 + beta*E2(z))
```

Esta es una ecuación implícita en `E2`: combinada con la definición de `E2`
es una **cuadrática exacta**. Con `A = OL*x^n*(1+beta)` y `B = Or*(1+z)^4 + m`:

```
beta*E2^2 + (1 - beta*B)*E2 - (A + B) = 0

E2 = [ beta*B - 1 + sqrt( (1-beta*B)^2 + 4*beta*(A+B) ) ] / (2*beta)
vacuum = A / (1 + beta*E2)
```

(square root positiva; `beta -> 0` recupera `rho_L = OL*x^n`).

### 2.3 La ecuación de acoplamiento materia-vacío (la que causa el problema)

La conservación de la energía impone una ODE para `m(z)`. Tal como está
implementada (idéntica en la referencia Python
`dvch_vs_lcdm_real.py`, en `dvch_camb_background.py` y en el Fortran del CAMB
parcheado `DVCHModel.f90`, las tres verificadas consistentes entre sí):

```
dvac_dm  = vacuum * n / m
dvac_de2 = -OL * x^n * (1+beta) * beta / (1 + beta*E2)^2
denom    = 1 - dvac_de2
coeff    = (1 + dvac_dm) * (1 + dvac_de2/denom)
explicit = 4*Or*(1+z)^3

dm/dz = [ 3*m/(1+z) - (dvac_de2/denom)*explicit ] / coeff
```

con `m(0) = Om`. Esta ODE dice, en esencia: cuando el vacío cambia con el
tiempo, la materia no sigue `a^-3`, sino que absorbe la transferencia.

### 2.4 Perturbaciones (cómo se metió en CAMB)

El fondo (`H(z)`, `grho`) se reemplaza por la integración de la ODE anterior.
La perturbación del CDM recibe una fuente de interacción (cierre "vacío en su
sistema de reposo", `delta rho_L = 0`):

```
qtilde     = -E*OL_z*bracket / (1 + n*OL_z/m)
bracket    = n - beta*(4*Or + 3*m) / (3*(1+beta*E2))

clxcdot = -k*z  +  3*(adotoa/(E*m)) * ( qtilde*clxc - delta_qtilde )
```

(con `delta_qtilde = c_m*m*delta_m + c_E*delta_E`, `delta_E = 0` en este
cierre; las `c_i` son derivadas analíticas de `qtilde`).

### 2.5 Variante "switch de desacoplamiento" (lo último que se probó)

Para separar el efecto temprano del tardío se introdujo un factor que apaga
la desviación del vacío respecto a LCDM a `z > z_c`:

```
S(z) = 1 / ( 1 + ((1+z)/(1+z_c))^p )        (env DVCH_ZC, DVCH_P; p=10)

vac = OL + S*(vac_raw - OL)     # vac_raw = cerradura original
```

La cuadrática sigue exacta con `C = Or*(1+z)^4 + m + OL*(1-S)` y
`K = S*A`:

```
beta*E2^2 + (1-beta*C)*E2 - (C+K) = 0
```

y la ODE escala `dvac_dm, dvac_de2 -> S*(...)`. Cuando `S -> 0` el modelo es
**exactamente LCDM** (verificado bit-idéntico en CAMB).

---

## 3. El problema, explicado

### 3.1 El déficit de materia fosilizado

Integrando la ODE desde `z=0` (con `m(0)=Om`) se obtiene la fracción

```
f(z) = m(z) / ( Om*(1+z)^3 )        # vale 1 en LCDM
```

**Resultado (reproducible con `_diag_ode_wide.py`):**

| Om | n | beta | f(z=1) | f(z=10) | f(z=1000) |
|----|---|------|--------|---------|-----------|
| 0.3021 | 0.518 | 1.104 | 0.665 | 0.707 | 0.707 |
| 0.3800 | 0.4226 | 0.5257 | 0.787 | 0.832 | 0.832 |
| 0.3800 | 0.3239 | 0.2894 (switch z_c=1) | 0.808 | 0.852 | 0.852 |
| 0.3021 | 0.097 | 0.101 | 0.933 | 0.987 | 0.987 |
| 0.3500 | 0.200 | 0.300 | 0.946 | 1.007 | 1.007 |
| 0.3150 | 0.01 | 0.01 | 0.995 | 1.012 | 1.012 |

Mecanismo (por qué el déficit no se recupera): a alto `z` el cociente
`vac/m ~ x^n/(beta*m) ~ (1+z)^(3n-3)` decae (porque `n < 1`), los términos
`dvac_*` se apagan y la ODE tiende a `dm/dz = 3m/(1+z)`. Es decir, **la
violación de `a^-3` se comete a `z <~ 3-10` (donde el acoplamiento aún es
grande) y a partir de ahí la fracción `f` se congela para siempre**. Por eso el
daño es permanente y no un "blip" transitorio.

### 3.2 Consecuencia para el CMB (verificado con CAMB parcheado)

Con `Om=0.3021, n=0.518, beta=1.104` (mediana tardía en cosmología del CMB):

```
H(z):  -8.8% (z=1),  -15% (z=2),  -15% ... -12% (z=10 ... 1000)
theta*:  104.12  ->  99.23     (-4.7%)   [Planck mide theta* a 1e-3]
r_drag:  147.1   ->  158.6     (+7.7%)
pico TT: 220 -> 224 ;  amplitud TT +8% (l=220) hasta +29% (l=1500)
```

chi2 del CMB (punto fijo, cosmología del mejor ajuste LCDM del CMB, nuisance
oficiales):

```
LCDM (n=0, beta=0):        2772.82
DVCH (mediana tardia):    61961   ->  Delta chi2 = +5.9e4
```

El `+5.9e4` **no** es un bug: la fuente de perturbación del CDM se desactivó
como prueba y el `+5.9e4` se mantuvo prácticamente igual (`+59170`); el origen
es 100% el fondo (la `H(z)` y la física de la era de materia).

### 3.3 Por qué la mejora tardía y el CMB son incompatibles

El ajuste tardío (SNe + BAO + `H(z)`) necesita desviación de LCDM a `z < 2.33`
... pero el déficit de materia se genera en esa misma ventana (`z < 3`).
Desconectar el vacío a `z > z_c` no sirve porque el daño ya está hecho:

| z_c | Delta chi2 tardío | Delta chi2 CMB |
|-----|-------------------|----------------|
| 1.0 | -3.92 (n=0.324, beta=0.289, Om=0.379) | +5.4e4 |
| 1.5 | -3.55 | +6.4e4 |
| 2.0 | -3.58 | +6.5e4 |
| 5.0 | -3.60 | +6.1e4 |
| sin switch | -3.60 (n=0.423, beta=0.526, Om=0.380) | +5.9e4 |

(`dvch_switch_tradeoff.csv`; el CMB se evaluó en cada mejor ajuste tardío del
`z_c` correspondiente.)

---

## 4. Todo lo que se probó (historial de esta auditoría)

1. **Corrección de datos**: DESI DR1 mal copiado (tabla 1 del paper),
   errorres en los 31 puntos de cosmic chronometers, uso indebido de
   Pantheon+ (calibradores incluidos en el ajuste, cortes por calidad),
   marginalización de `r_d` en vez de fijarlo. Todo re-hecho desde cero.
2. **Física**: el background original violaba la conservación de la energía;
   se derivó e implementó la ODE exacta (sección 2.3).
3. **Late-time rerun**: Pantheon+ (1580 SNe) + DESI DR1 (12 BAO) + 31 CC:
   `chi2`: LCDM 1416.51, `w0wa`CDM 1413.16, DVCH 1412.91.
   `Delta chi2 = -3.60`, `Delta AIC = +0.4` (el modelo **no** gana).
4. **Evidence (dynesty)**: `ln Z`: LCDM -722.32, `w0wa` -725.47, DVCH -722.50.
   `Delta ln Z = -0.18 +- 0.49` (evidence nula; lasclaimas originales de
   `Delta ln Z ~ +10` eran por likelihoods mal escritas).
5. **CMB (CAMB parcheado)**: minimizadores con nuisance oficiales libres:
   LCDM `chi2=2766.81`, DVCH `chi2=2780.85` (n=0.097, beta=0.101),
   `Delta chi2 = +14.04`, `p ~ 9e-4`. El mejor ajuste del CMB cae en
   `n ~ 0.1` (la frontera del prior, es decir "lo más parecido a LCDM posible
   sin ser LCDM").
6. **Aislamiento del origen**: apagar la fuente de perturbación del CDM
   (no cambia el `+5.9e4`); comparar CAMB vs referencia Python vs integración
   RK4 propia hasta `z=1100` (concuerdan a `~1e-3`); comparar espectros y
   parámetros derivados (`theta*`, `r_drag`, picos TT).
7. **Variante switch de desacoplamiento** (sección 2.5): 7 valores de `z_c`,
   trade-off completo de la sección 3.3. Ninguno resucita el modelo.
8. **Tests de anidamiento**: `n=0, beta=0` reproduce LCDM bit-idéntico
   (`chi2 = 2772.82`), tanto con switch apagado como con switch en `z_c`.

---

## 5. Condiciones necesarias para una hipótesis viable

Si alguien quiere construir una fórmula que mantenga el espíritu (vacío que
intercambia energía con la materia) pero que sobreviva los datos, estas son
las restricciones cuantitativas que debe cumplir:

1. **Era de materia:** `f(z) = m(z)/(Om*(1+z)^3)` debe tender a 1 (no a una
   constante `c < 1`) para `z >~ 3`. Si se satura en `c`, el CMB recibe un
   `Delta H/H ~ -(1-c)/2` del orden del -(5..15)%, i.e. `theta*` se desplaza
   varios % y `chi2 ~ 1e4+`.
   - El acoplamiento debe apagarse **antes** de generar un déficit apreciable,
     i.e. a `z <~ 0.5-1` (y entonces hay que ganar el `Delta chi2` tardío solo
     con la dinámica de `z < 0.5`).
2. **`theta*` y `r_d`**: tolerancia `~0.1%` en `H(z)` acumulada entre `z=0` y
   `z=1100` (Planck mide `theta*` a `1e-3`). Eso implica que la desviación de
   `H(z)` respecto a LCDM debe ser `< 0.3%` en todo `1 < z < 1100` si `Om` y
   `H0` son los de Planck.
3. **`Om`**: el ajuste tardío actual empuja `Om ~ 0.38` (frente a
   `0.315 +- 0.007` de Planck). Una alternativa viable debe o bien no empujar
   `Om`, o la mejora tardía debe ser muy grande para compensar la tensión.
4. **Perturbaciones**: hay que especificar el cierre perturbativo
   (rest-frame del vacío, `delta rho_L = 0` no es necesariamente la opción
   correcta) y pasar el CMB incluyendo lensing e ISW.
5. **Ganancia real**: `w0wa`CDM (2 parámetros extra) da
   `Delta chi2 = -3.35` en los mismos datos; cualquier modelo con >= 2
   parámetros extra debe mejorar eso para ser interesante.

**Sugerencias de diseño** (no probadas): acoplamiento proporcional a
`(rho_L/rho_m)^k` con `k` grande (muere en la era de materia), o cutoff
explícito `(1+z)^{-m}`; o vacío que solo empieza a desviarse bajo un umbral de
densidad; o usar la misma idea pero con `n > 1`?? (en esta familia `n < 1`
hace decaer el acoplamiento, pero es justo lo que permite el déficit a bajo
`z`).

---

## 6. Números de referencia del pipeline actual

- Late-time: `chi2` LCDM 1416.51 / `w0wa` 1413.16 / DVCH 1412.91;
  `Delta AIC = +0.4`; `ln Z`: -722.32 / -725.47 / -722.50.
- CMB minimizers: LCDM 2766.81 / DVCH 2780.85 (n=0.097, beta=0.101);
  `Delta chi2 = +14.04`, `p ~ 9e-4`.
- CMB punto fijo (cosmología LCDM del CMB, `H0=67.36, ombh2=0.02237,
  omch2=0.1200, tau=0.0544, As=2.1e-9, ns=0.965, mnu=0.06`):
  LCDM 2772.82, DVCH (mediana tardía) 61961.

## 7. Archivos para reproducir

- `dvch_vs_lcdm_real.py` — likelihood tardía + cerradura Python (grid `z<=2.61`).
- `dvch_camb_background.py` — referencia de la ODE (DOP853).
- `camb_patch/DVCHModel.f90`, `camb_patch/README_patch.md` — parche CAMB.
- `cmb_npoint_check.py` — chi2 CMB en un punto (`DVCH_N`/`DVCH_BETA`
  sobreescriben `n`, `beta`).
- `switch_tradeoff.py`, `dvch_switch_tradeoff.csv` — trade-off tardío.
- `_cmb_tradeoff.sh` — trade-off CMB por `z_c`.
- `_diag_ode_wide.py` — integración de la ODE hasta `z=1100` (la que
  reveló el déficit fosilizado).
- `_diag_cmb.py` — espectros y parámetros derivados de CAMB.
- `_diag_pathology.py` — mapa del déficit en el espacio de parámetros.
- `SESSION_LOG.txt` sección 7 — resumen de este diagnóstico.
