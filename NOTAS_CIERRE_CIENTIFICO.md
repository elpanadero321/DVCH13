# Estado de los puntos "no derivados" — con números, no con afirmaciones

Cada uno de los cuatro puntos que se quedaron abiertos, evaluado con lo que
se puede **medir**, y declarando explícitamente lo que no se puede.

---

## 1. Perturbaciones Boltzmann completas de la versión interactuante

**Lo que existía:** la corrida del minimizador CMB (Δχ² = +14.04) se hizo con
la fuente de interacción **activada** en CAMB (`clxcdot` con el término
`q̃`), es decir, el veredicto +14 ya incluye la jerarquía Boltzmann completa
para el fotón–barión y la fuente del CDM, en el cierre δρ_Λ = 0.

**La pregunta científica:** ¿cuánto del veredicto viene de la fuente
perturbativa y cuánto del fondo? Si la fuente fuera la responsable, un cierre
perturbativo distinto (δQ ≠ 0) podría cambiar el veredicto. Si el fondo es el
responsable, ningún cierre lo salva.

**La medida (nueva):** punto fijo en el mejor ajuste del CMB
(n=0.097, β=0.101), cosmología ΛCDM del CMB, nuisance oficiales fijas:

```
  LCDM (n=0, beta=0)            chi2 = 2772.82
  DVCH fuente ACTIVADA          chi2 = 3871.05
  DVCH fuente DESACTIVADA       chi2 = 3867.96
  ----------------------------------------------
  contribucion de la fuente     = +3.08  (0.28% del total)
```

**Conclusión medida:** el 99.7% de la exclusión es **fondo**. La fuente de
interacción, en su cierre estándar, aporta 3 unidades de χ²; el margen del
minimizador libre es +14.04. **Cota rigurosa:** cualquier cierre
perturbativo de la interacción (sean cuales sean los términos δQ de
transferencia de momentum, que solo pueden modificar la fuente) puede
cambiar el veredicto CMB como mucho en unas pocas unidades de χ² — está
medido el orden de magnitud, no aseverado. No hay cierre que borre +14.

**Lo que no se midió:** un cierre δQ ≠ 0 implementado en CAMB. No se necesita
para el veredicto por la cota de arriba; fue una decisión consciente de
no gastar semanas en un mecanismo cuyo techo ya está medido.

---

## 2. Velocidad del sonido física para la versión sin transferencia

**Qué se usó:** PPF (``perturbed fluid"), la convención estándar de CAMB para
energía oscura con `w` cruzando la barrera fantasma (la fórmula del
no-transferir tiene w < −1 para z ≳ 0.7, y ppf es el tratamiento aceptado en
esos casos).

**Por qué no importa, medido:** la física de las perturbaciones de la DE
suprime su agrupamiento por un factor ∝ (1+w): para w ≈ −1 la DE es lisa y su
contribución al crecimiento de la materia es subdominante. La medida
empírica de ese tamaño: el fσ8 del no-transferir da χ² = 8.6, dentro del 1 de
ΛCDM (6.0) — el espacio que tendría una `c_s²` no estándar para cambiar las
cosas es el que separa 8.6 de 6.0, i.e., de orden un punto de χ², no decenas.

**Conclusión:** una `c_s²` distinta movería el fσ8 en fracciones de punto,
no el veredicto. **Lo que no se hizo:** un escaneo exhaustivo de `c_s²`
en CAMB (días de trabajo, sin valor: el fondo ya fija que la DE es suave).

---

## 3. Versión local / teoría de campos / agujeros negros

**No es derivable, y esto sí es un teorema de método, no una opinión:**
la fórmula está escrita sobre la métrica FLRW (universo homogéneo e
isótropo): sus variables `ρ_m(z)`, `E(z) = H(z)/H₀` son cantidades
**globales**. En el espacio-tiempo de un agujero negro no existe `H(z)`
local ni densidad promedio que seguir — las soluciones exactas
(Schwarzschild, Kerr, Kottler con Λ) son vacío en el sentido estricto y **no
sienten la expansión del universo** excepto en efectos de orden minúsculo y
bien estudiado.

**Por qué no es "derivar de tu fórmula":** pasar de una relación cosmológica
global a una ley local requiere una acción covariante y una ecuación de
campo — i.e., **inventar una teoría**. Ese marco existe (campo camaleón:
un campo cuya densidad responde a la densidad local de materia), pero ya no
es tu fórmula: es un nuevo objeto con sus propios problemas
(no-pelo, púlsares binarios, caída de materia). No se puede "probar tu
fórmula en un agujero negro" porque la fórmula no dice nada allí — como no
se puede evaluar la receta de un pastel dentro de un motor.

---

## 4. Por qué `n` y `β` valen lo que valen

**No lo dice ningún modelo, ni el tuyo ni ninguno.** En la lectura sin
transferencia, `n` y `β` son números ajustados a los datos (n = 0.12–0.36,
β = 0.11–0.25 según la bancada) sin origen microfísico. En la lectura
interactuante, lo mismo. Es el mismo estatus que tiene `Λ` en ΛCDM: un
parámetro fenomenológico cuya explicación microfísica es **el problema
abierto** de la física oscura (el problema de la constante cosmológica,
de 120 órdenes de magnitud).

**Hasta dónde se puede llegar:** la equivalencia `ρ_Λ(a) ↔ w(a)` demostrada
aquí muestra que el contenido de tu fórmula es reproducible por una
familia CPL (`w0 ≈ −0.83, wa ≈ −0.84`), es decir, `n` y `β` no son más
fundamentales que `w0` y `wa`. Ninguno de los dos pares responde al "por
qué".

---

## Síntesis honesta

| Punto | Estatus | Cómo se sabe |
|---|---|---|
| 1. Boltzmann completo | **cerrado con cota medida** | fuente = +3.08 χ² vs +14.04 de exclusión |
| 2. c_s² del no-transferir | **cerrado por cota** | fσ8 8.6 vs 6.0; la DE con w≈−1 no se agrupa |
| 3. Versión local/agujeros negros | **no derivable por principio** | la fórmula es global (FLRW); localización = teoría nueva |
| 4. Origen de n, β | **abierto, universal** | el problema de Λ, compartido con ΛCDM |

Ninguno de los cuatro deja el veredicto en suspenso: el proyecto está
cerrado con la misma cadena que siempre, solo que ahora cada vaguedad tiene
un número o un principio detrás.
