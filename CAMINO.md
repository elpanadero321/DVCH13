# El camino — cómo la hipótesis del vacío unido a la materia llegó a su veredicto

Este documento no es una lista de χ². Es la **cadena de razonamiento** del
proyecto: cómo se pasó de la intuición original a un veredicto completo y
reproducible. Cada paso explica *por qué* se dio y qué se aprendió.

---

## 1. La idea original

**Intuición:** la energía oscura y la materia no son dos cosas separadas. Son
dos caras de un mismo sector: donde hay materia, el vacío se manifiesta de
otra forma. La materia oscura, en esta visión, no es una partícula nueva sino
el vacío "concentrado" o "seguido" a la materia.

**Fórmula del paper:**

```
ρ_vacío(z) = Ω_Λ · (ρ_materia(z)/ρ_materia(hoy))^n · (1+β) / (1+β·E²)
```

En palabras: la densidad del vacío no es constante, sino que **crece con la
materia** (el término `(ρ_materia)^n`) y, además, lleva un **freno**
(`1/(1+βE²)`) que la apaga cuando el universo estaba muy denso.

**Lo que la fórmula quiere resolver:** la "coincidencia cósmica" — por qué la
energía oscura y la materia son comparables justamente ahora. Si el vacío
sigue a la materia, nunca dejan de ser comparables.

---

## 2. La auditoría encuentra el primer problema: la conservación

La fórmula, tal como estaba en el paper, **no cerraba el balance de energía**.
La densidad del vacío cambiaba con el tiempo, pero el paper no decía de dónde
venía o adónde iba esa energía. Era como una cuenta bancaria cuyo saldo
cambia solo.

**El arreglo físico:** añadir la ecuación de balance (conservación de la
energía total de los dos fluidos):

```
d(ρ_materia a³)/dt = −a³ · d(ρ_vacío)/dt
```

Es decir: cada joule que el vacío pierde, la materia lo gana. A esa
transferencia la llamamos **Q**.

**Lección:** una ecuación aislada puede ser bonita, pero la física la tienen
las ecuaciones que la rodean. La conservación no es un adorno burocrático: es
la que convierte una fórmula en un modelo.

---

## 3. La consecuencia inevitable: el déficit fosilizado

Una vez añadido el balance, el modelo dice que la materia oscura recibe
energía del vacío **a lo largo de toda la historia**. Y aquí salió el efecto
que mató todo:

**La materia creada llega "en reposo".** Imagina un universo donde, en cada
punto del espacio, aparece continuamente un poco de materia oscura nueva,
sin velocidad. En una región donde la materia está un poco más agrupada (la
semilla de una galaxia), esa materia nueva no se suma al movimiento: lo
**diluye**. El agrupamiento se frena.

El cálculo exacto, integrando la ecuación de conservación desde hoy hacia
atrás, da:

```
f(z) = ρ_materia(z) / (Ω_m(1+z)³)   ← vale 1 en el modelo estándar
en la región de parámetros que ajusta las supernovas:  f → 0.83 … 0.71
y, una vez que llega ahí, SE CONGELA para siempre (a z=100, z=1100, igual)
```

A ese fenómeno lo llamamos **déficit fosilizado**: el daño se hace a bajo
corrimiento al rojo (z < 10) y queda grabado en la historia del universo para
siempre. Ni el freno `β` de la fórmula lo evita — porque el freno actúa sobre
la *densidad* del vacío, no sobre la *transferencia* ya realizada.

**Lección:** en cosmología, el pasado no se borra. Una formula que toca la materia
a bajo z deja una huella permanente que el CMB leerá sin piedad.

---

## 4. Los tres bancos de datos — y la frontera

El modelo se midió contra todos los datos disponibles, por separado y juntos:

- **Expansión tardía** (supernovas Pantheon+, BAO de DESI, H(z) directos):
  el modelo **mejoraba un poco** (Δχ² ≈ −3.6). La intuición de que "el vacío
  unido a la materia" ayuda a explicar la aceleración **no estaba loca**:
  ajustaba el universo actual tan bien como ΛCDM.
- **CMB** (Planck): el déficit fosilizado significaba que en la recombinación
  había un 15–30% menos de materia y una expansión 8–12% más rápida. Planck
  mide la escala acústica con 0.03% de precisión: el modelo fracasaba por un
  factor enorme (Δχ² del orden de 10⁴–10⁵).
- **Formación de estructura** (velocidades de galaxias, fσ8): la dilución de
  las perturbaciones daba un crecimiento 20–25% menor del observado. Δχ²
  ≈ 486 contra 6 de ΛCDM.

**El patrón clave — el trade-off:** la cantidad de transferencia que la
expansión necesita es justamente la que destruye la estructura. Se midió la
curva completa: para cada valor del parámetro, ambas cosas pasan a la vez.
**No hay ventana.**

**Lección:** un modelo nuevo se prueba en TODOS los datos, no solo en los que
lo favorecen. La expansión lo aprobaba; el CMB y las galaxias lo reprobaban.
Y la física del modelo no permite separar las dos cosas — es la misma Q
haciéndolas.

---

## 5. Varias pruebas de rescue — todas medidas

Antes del veredicto final, se probaron todos los caminos que la propia
hipótesis permitía:

| Variante | Idea | Veredicto |
|---|---|---|
| switch de desacoplamiento | apagar el vacío a alto z | el daño ya estaba hecho a bajo z |
| generalización `a1(1−a)+a2(1−a)²` | familia completa de formas | la trade-off persists: no hay ventana |
| sólo pegamento (fuerza, no energía) | acoplamiento de momentum | consistente, pero sin detección (1σ) |
| vacío que crece (fantasma) | la DE aumenta sola | pierde la expansión y la estructura |
| DE que varía sola | el "agua" que cambia de fase | consistente — pero es física estándar |
| **fórmula sin transferencia** | quitar la ecuación de balance | **funciona** — ver §6 |

**Lección:** la exclusión no fue apresurada ni se basó en una sola
formulación. Se midieron todas las versiones que la hipótesis permitía, una
a una, con la misma pipeline.

---

## 6. La resolución final: quitar la transferencia

El usuario insistió (con razón) en una última posibilidad: **¿y si la
fórmula simplemente no transfiere?** Es decir: materia `a⁻³` normal, vacío
evolucionando según su propia fórmula, sin ecuación de balance.

El resultado: **funciona.** Pasa los cuatro bancos:

- expansión tardía: Δχ² = −3.6 (mejora marginal)
- CMB: desviación 0.00% a z > 100 (no se rompe)
- formación de estructura: χ² = 8.6 ≈ ΛCDM (6.0)
- conjunto completo: Δχ² = −2.36 con 2 parámetros (~1.5σ: **no
  significativo**)

**Pero esto tiene un precio conceptual, y es el punto final del proyecto:**
sin la ecuación de balance, **el vacío deja de saber que la materia existe**.
El "seguimiento" entre ambos deja de ser una interacción y se convierte en
una coincidencia del ajuste. La fórmula se transforma matemáticamente en un
perfil `w(a)`: energía oscura evolutiva, `w0 ≈ −0.83, wa ≈ −0.84` — la misma
familia que el `w0waCDM` estándar que la cosmología usa desde 1998.

**Y la ironía final:** sin la ecuacion de balance, la energia del vacio vendria de
la nada (violación de conservación — el problema original). La única forma
de salvarla es convertirla en un fluido autónomo — y entonces deja de ser la
idea del paper. La semilla que motivó el trabajo es, precisamente, la pieza
que los datos prohíben.

---

## 7. Qué quedó de todo esto

1. **La auditoría:** los claims originales del paper (Δln Z ≈ +10, >5σ) eran
   producidos por likelihoods mal escritas. Corregir los datos y las
   estadísticas fue la primera mitad del trabajo, y es válida
   independientemente de todo lo demás.
2. **La exclusión cuantitativa de la interacción** — no una opinión, sino una
   cadena completa: déficit fosilizado → CMB + estructura. Reproducible,
   validada contra CAMB (el solver de crecimiento coincide bit a bit con el
   código de Boltzmann profesional).
3. **La fórmula sin transferencia** sobrevive como parameterización de DE
   evolutiva — física estándar, sin descubrimiento, pero consistente.
4. **La metodología:** una pipeline completa (likelihoods correctas,
   marginalización de r_d, tests de mutación, validación cruzada contra
   CAMB, ajuste conjunto CMB+BAO+SNe con nuisance oficial libre). Esa
   herramienta sirve para la próxima pregunta.

---

## 8. La pregunta que sigue abierta

**¿Por qué existe la energía oscura y por qué es tan débil?** La QFT del
vacío predice 10¹²⁰ veces más de lo que se observa. Ese problema sigue
abierto, legítimo, y es justo lo que el paper intentaba atacar. El proyecto
no lo resolvió — pero cerró uno de los caminos posibles con números, y eso
también es avance.

La frontera real de la física oscura hoy no está en "vacío interactuando con
materia" (medido y excluido), sino en:
- microfísica de la materia oscura a escala enana (perfiles de halos,
  satélites),
- la naturaleza cuántica de la energía oscura (campos, condensados),
- la tensión H0 y σ8 — anomalías reales pero de 2–3σ.

Ahí hay espacio. En lo que este proyecto midió, no.

---

*Documento reproducible: todos los números de esta cadena están en
`SESSION_LOG.txt` y re-generables con los scripts del repositorio
(`dvch_vs_lcdm_real.py`, `alpha_general_fit.py`, `run_notrans_joint.py`,
`switch_tradeoff.py`, y la carpeta `tests_aux/`).*
