"""Funciones de likelihood/prior a nivel de modulo para muestreo anidado.

dynesty necesita funciones picklables (Windows usa spawn), de modo que las
posteriores de los tres modelos se definen aqui a nivel de modulo y el
pool de procesos las importa directamente.
"""
import numpy as np

import dvch_vs_lcdm_real as m
import w0wa_control as wc

SNE = m.load_pantheon()

BOUNDS = {
    "lcdm": m.BOUNDS["lcdm"],
    "w0wa": wc.BOUNDS,
    "dvch": m.BOUNDS["dvch"],
}

_LOGPOST = {}


def _make(model):
    if model == "w0wa":
        def logpost(p):
            Om, H0, M, w0, wa, rd = p
            for (lo, hi), v in zip(BOUNDS[model], p):
                if not (lo < v < hi):
                    return -np.inf
            c2 = wc.chi2_all_w0wa(Om, H0, M, w0, wa, rd)
            if c2 > 1e9:
                return -np.inf
            return -0.5 * c2
    else:
        logpost = m.make_logpost(model, SNE)
    return logpost


for _model in ("lcdm", "w0wa", "dvch"):
    _LOGPOST[_model] = _make(_model)


def logpost_lcdm(p):
    return _LOGPOST["lcdm"](p)


def logpost_w0wa(p):
    return _LOGPOST["w0wa"](p)


def logpost_dvch(p):
    return _LOGPOST["dvch"](p)


def ptform(u, model):
    return np.array([lo + (hi - lo) * ui
                     for ui, (lo, hi) in zip(u, BOUNDS[model])])


def ptform_lcdm(u):
    return ptform(u, "lcdm")


def ptform_w0wa(u):
    return ptform(u, "w0wa")


def ptform_dvch(u):
    return ptform(u, "dvch")
