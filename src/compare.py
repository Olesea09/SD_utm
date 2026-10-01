"""Pasul 6 al pipeline-ului: vectorul de comparare al fiecărei perechi candidate.

Pentru fiecare pereche (rândul a, produsul b din Nomenclator) se calculează cât de asemănătoare sunt
câmpurile, în două variante:
  - scoruri continue între 0 și 1 (sim_*, eq_*), pentru regresia logistică, SVM și rețeaua neuronală;
  - niveluri discrete (niv_*), pentru modelul Fellegi–Sunter, care lucrează cu categorii de acord
    („identic”, „asemănător”, „diferit”).
O valoare lipsă în oricare dintre surse dă NaN (respectiv <NA> la niveluri): lipsa nu e nici acord, nici dezacord.
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# Pragurile nivelurilor (alese pe date: vezi notebook 03, secțiunea 5)
NAME_LEVELS = (0.92, 0.80)   # Jaro–Winkler: >= 0.92 „foarte asemănător”, >= 0.80 „asemănător”
FIRM_LEVELS = (0.90, 0.60)   # token set ratio pe numele firmei
FORM_SIMILAR = 0.5           # proporția de cuvinte comune la formă


def _tokens(x, sep=r"[\s/+]+"):
    return set(t for t in re.split(sep, x) if t) if isinstance(x, str) else None


def _dose_parts(x):
    """Componentele cu cifre ale dozei („25 mg/ml” -> {„25 mg”}); „ml” singur nu e o componentă."""
    t = _tokens(x, r"/")
    return None if t is None else {p for p in t if re.search(r"\d", p)}


def _jaccard(x, y):
    if x is None or y is None:
        return np.nan
    return len(x & y) / len(x | y) if x | y else np.nan


def _pairwise(fn, xs, ys):
    return np.array([fn(x, y) if isinstance(x, str) and isinstance(y, str) else np.nan
                     for x, y in zip(xs, ys)], dtype=float)


def _firm_sim(x, y):
    """Cea mai mare asemănare între oricare firmă a lui a și oricare firmă a lui b (deținători și producători)."""
    return max(fuzz.token_set_ratio(p, q) for p in x.split("|") for q in y.split("|")) / 100


def _eq(x, y):
    """1 dacă valorile sunt egale, 0 dacă diferă, NaN dacă una lipsește."""
    out = (x.values == y.values).astype(float)
    out[pd.isna(x.values) | pd.isna(y.values)] = np.nan
    return out


def _levels(values, cuts, top_exact=None):
    """Niveluri ordonate: cel mai mare pentru identic (dacă top_exact), apoi pe praguri descrescătoare."""
    v = np.asarray(values, dtype=float)
    lvl = np.zeros(len(v))
    for i, c in enumerate(sorted(cuts)):
        lvl[v >= c] = i + 1
    if top_exact is not None:
        lvl[v >= 1.0 - 1e-12] = len(cuts) + 1
    return pd.array(np.where(np.isnan(v), pd.NA, lvl), dtype="Int64")


def _three(eq, similar):
    """2 = egal, 1 = asemănător, 0 = diferit, <NA> = lipsește în una dintre surse."""
    lvl = np.where(eq == 1, 2, np.where(similar, 1, 0)).astype(float)
    lvl[np.isnan(np.asarray(eq, dtype=float))] = np.nan
    return pd.array(np.where(np.isnan(lvl), pd.NA, lvl), dtype="Int64")


def compare(pairs: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Vectorul de comparare pentru perechile (ia, ib). a și b au coloanele normalizate din pașii 3–4."""
    A = a.loc[pairs.ia].reset_index(drop=True)
    B = b.loc[pairs.ib].reset_index(drop=True)
    v = pairs[["ia", "ib"]].reset_index(drop=True).copy()

    # denumirea: Jaro–Winkler (greșeli de litere) și token set (cuvinte în plus sau lipsă)
    v["sim_denumire"] = _pairwise(JaroWinkler.similarity, A.denumire_norm, B.denumire_norm)
    v["sim_denumire_tok"] = _pairwise(lambda x, y: fuzz.token_set_ratio(x, y) / 100, A.denumire_norm, B.denumire_norm)
    # doza: egalitate și proporția de componente comune („10000 ui” vs „400 ui/7200 ui/10000 ui” -> 1/3)
    v["eq_doza"] = _eq(A.doza_norm, B.doza_norm)
    v["sim_doza"] = [_jaccard(_dose_parts(x), _dose_parts(y)) for x, y in zip(A.doza_norm, B.doza_norm)]
    # forma: egalitate și proporția de cuvinte comune („comprimate” vs „comprimate filmate” -> 1/2)
    v["eq_forma"] = _eq(A.forma_norm, B.forma_norm)
    v["sim_forma"] = [_jaccard(_tokens(x), _tokens(y)) for x, y in zip(A.forma_norm, B.forma_norm)]
    # divizarea: numărul total de unități și volumul, separat
    v["eq_unitati"] = _eq(A.unitati_norm, B.unitati_norm)
    v["eq_volum"] = _eq(A.volum_norm, B.volum_norm)
    # firma: cea mai bună potrivire între deținători și producători
    v["sim_firma"] = _pairwise(_firm_sim, A.firme, B.firme)
    # substanța activă (relevant la perechile aduse doar de prefixul denumirii)
    v["eq_dci"] = _eq(A.dci_stem, B.dci_stem)

    # niveluri pentru Fellegi–Sunter
    v["niv_denumire"] = _levels(v.sim_denumire, NAME_LEVELS, top_exact=True)            # 0..3
    v["niv_doza"] = _three(v.eq_doza, v.sim_doza > 0)                                     # 0 diferit, 1 parțial, 2 egal
    v["niv_forma"] = _three(v.eq_forma, v.sim_forma >= FORM_SIMILAR)                      # 0, 1 asemănător, 2 egal
    v["niv_unitati"] = pd.array(v.eq_unitati, dtype="Int64")                              # 0 / 1
    v["niv_volum"] = pd.array(v.eq_volum, dtype="Int64")                                  # 0 / 1
    v["niv_firma"] = _levels(v.sim_firma, FIRM_LEVELS)                                    # 0..2
    v["niv_dci"] = pd.array(v.eq_dci, dtype="Int64")                                      # 0 / 1
    return v


SIM_COLS = ["sim_denumire", "sim_denumire_tok", "eq_doza", "sim_doza", "eq_forma", "sim_forma",
            "eq_unitati", "eq_volum", "sim_firma", "eq_dci"]
LEVEL_COLS = ["niv_denumire", "niv_doza", "niv_forma", "niv_unitati", "niv_volum", "niv_firma", "niv_dci"]
