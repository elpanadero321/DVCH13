# Modelo α — qué funciona, qué falla y qué falta (handoff)

## La fórmula (ya verificada por dos IA de forma independiente)

```
E2(a) = Or0*a^-4 + Om0*a^-3 + OL0 + alpha*(1-a)^2          (a = 1/(1+z))
rho_c(a) = Oc0*a^-3 + (2*alpha/3)*a*(1-a)                  (CDM; bariones a^-3)
rho_L(a) = OL0 + (alpha/3)*(1-a)*(3-5*a)
Q = (2*alpha/3)*H*rho_crit0*a*(4-5a)                       (rho_m+3Hrho_m=Q, rho_L=-Q)
alpha=0 == LCDM exacto
```

## ✓ Qué está arreglado (vs. modelo DVCH original)

- **Cierre matemático**: identidad de Friedmann a 1e-15, conservación exacta,
  normalización en a→0 y a→1 (verificado con derivadas analíticas y con ODE
  independiente; script `alpha_model_fit.py`, y el script externo
  `D:\verificar_cierre_vacio.py`).
- **No hay déficit de materia fosilizado**: la desviación del CDM respecto a
  `a^-3` es `(2*alpha/(3*Oc0))*a^4*(1-a)` → va a 0 en el universo temprano.
  `dH/H(z=1100) ~ 8e-10` (el modelo original daba −12%…−15%: el infierno del
  CMB). Esto era el problema estructural que mató al modelo (n, beta).

## ✗ Qué falla ahora (ajuste tardío, datos reales)

Pantheon+ (1580 SNe, z>0.01 sin calibradoras) + DESI DR1 (12 BAO) + 31 cosmic
chronometers, `r_d` marginalizado, mismas likelihoods que el pipeline:

| modelo | params | chi2 | Δchi2 vs LCDM |
|---|---|---|---|
| LCDM | 4 | 1416.51 | — |
| α-modelo | 5 | 1415.10 | **−1.41** |
| w0waCDM | 6 | 1413.16 | −3.35 |
| DVCH (n,β) | 6 | 1412.91 | −3.60 (pero el CMB lo destruye) |

Mejor ajuste: `Om=0.296, H0=68.2, alpha=0.597, rd=146.8`; MCMC:
`alpha = 0.607 +0.50 −0.50` → **solo ~1.2σ**. ΔAIC = +0.59 (LCDM gana por
parsimonia). **No mejora lo suficiente** para ser interesante por sí solo.
Asimismo la mejora va enα>0, donde el vacío era MÁS denso en el pasado
(`rho_L(a->0) = OL0 + alpha`), una constante tipo "early dark energy"
(despreciable dinámicamente a altas z, pero conceptualmente ad hoc).

En el mejor ajuste: `dH/H = +2.4% (z=1)`, `rho_c +13% (z≈0.25)` por la
transferencia vacuum→materia.

**Hallazgo nuevo (comprobado con CAMB estándar)**: θ* **no** está a salvo.
El horizonte sonoro `r_d = ∫_{z_d}^∞ c_s dz/H` solo involucra z > z_drag ≈
1060, donde el modelo es idéntico a ΛCDM; solo se mueve `D_A*`, así que el
cociente θ* = r*/D_A* no se cancela. A `wm_h2 = 0.1430` fijo:
Δθ*/θ* = +0.39% (α=0.3), **+0.78% (α=0.597)**, +1.29% (α=1.0). Planck mide
θ* al 0.03% ⇒ +0.78% ≈ 26σ. El CMB obliga a compensar bajando H0 / subiendo
Om (estilo `H0 ≈ 64.8, Om ≈ 0.34`), mientras el ajuste tardío da H0 = 68.2,
Om = 0.296. **El riesgo principal es este (fondo + geometría), no el ISW.**

**Equivalencia exacta**: el fondo del modelo α es una DE desacoplada con
`rho_DE = OL + alpha(1-a)^2` y
`w_eff(a) = −1 + 2α·a(1−a)/(3(OL + α(1−a)²))`, de modo que `w(a=1) = −1`
exacto (w_eff(z=1) = −0.88 con α=0.597; equivalente CPL wa ≈ +0.57).
Toda la familia `α(1−a)^m` con m > 1 tiene w0 = −1 por construcción y por eso
no puede llegar a w0wa. Ver `INSTRUCCIONES_PARA_IA.md` paso 5.

## ✗✗ Qué falta por completo (lo que hay que construir)

1. **Cierre de perturbaciones (lo más importante)**. El modelo está definido
   solo a nivel de fondo homogéneo. Para comparar con el CMB y con estructuras
   hace falta la teoría perturbativa del vacío que interactúa:
   - ecuación para `delta rho_L` (o decisión de cálculo: "vacuum rest frame"
     `delta rho_L = 0`, o vacío sin anisotropic stress `sigma_L = 0`, o
     derivación completa de la teoría de campos),
   - el término de transferencia de momento `delta Q` / `f_Q` y su split
     CDM/bariones,
   - gauge-invariancia y continuidad de la métrica de perturbación con
     CAMB/cobaya.
   El modelo DVCH anterior usaba el cierre `delta rho_L = 0`; si se reutiliza
     aquí, el término fuente del CDM sería proporcional a
     `Q_bar = (2α/3)*Hbar*a(4−5a)` y sus derivadas analíticas.
2. **Test CMB completo**: CAMB parcheado (fondo(alpha) + perturbación(alpha))
   + likelihood de Planck (TT/TE/EE/lensing). Puntos críticos a vigilar:
   - **`theta*`: confirmado que NO está a salvo** (+0.78% ≈ 26σ a α=0.597;
     ver arriba y `INSTRUCCIONES_PARA_IA.md`), hace falta un ajuste conjunto
     CMB+BAO+SNe con α libre;
   - **ISW tardío** (dH/H = +2.4% en z=1 es grande),
   - **lensing/σ8** (CDM +13% en z=0.25 modifica el crecimiento).
   Para que el modelo valga la pena, el CMB debe quedar con
   `Delta chi2 < ~ +5` en el mejor ajuste conjunto.
3. **Mecanismo físico para α**: hoy es un parámetro fenomenológico ad hoc; no
   hay derivación (teoría de campos, cutoff holográfico, etc.), ni explicación
   del cambio de signo de Q en a=0.8, ni por qué la corrección afecta solo al
   CDM y no a los bariones (asignación actual: decisión de diseño, no
   derivada).
4. **BBN / universo temprano**: la componente `OL0 + alpha` constante es
   dinámicamente despreciable en la era de radiación (ratio → 0), pero conviene
   verificar `N_eff`-like / BBN si α resulta grande.
5. **Si se quiere que mejore el ajuste**: con 1 parámetro solo llega a −1.41.
   Hay que pensar si el "ingrediente activo" es suficiente (por ejemplo
   generalizar `alpha*(1-a)^2` a `alpha*(1-a)^m` y ver si `m != 2` da más
   mejora), o aceptar que el modelo es un χ²-tie con LCDM.

## Archivos

- `alpha_model_fit.py` — verificación + ajuste tardío (reproduce todo).
- `dvch_alpha_fit.csv`, `dvch_alpha_chain.npy` — resultados.
- `DIAGNOSTICO_ESTRUCTURAL.md` — por qué el modelo original (n, β) falló.
- `SESSION_LOG.txt` secciones 7 y 8.
- Externo (de la otra IA): `D:\verificar_cierre_vacio.py`,
  `D:\verificacion_cierre_vacio.json`.
