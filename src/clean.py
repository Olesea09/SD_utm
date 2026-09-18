"""Pasul 2 al pipeline-ului: curățarea de BAZĂ (fără normalizare semantică).

Ce face:  elimină spațiile/tab-urile parazite, unifică valorile lipsă („undefined”, „-”, gol),
          calculează numărul de unități din ambalaj și prețul pe unitate.
Ce NU face (intră în pasul de normalizare, săptămâna 2): unificarea abrevierilor
          („comp. film.” -> „comprimate filmate”), conversia unităților de doză, traducerea RU->RO.
"""
import re
import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "-", "undefined", "nan", "none"}


def clean_text(s: pd.Series) -> pd.Series:
    """Spații multiple/tab-uri/NBSP -> un spațiu; capetele tăiate; token-uri de „lipsă” -> NaN."""
    out = (s.astype("string")
             .str.replace("\u00a0", " ", regex=False)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())
    return out.mask(out.str.lower().isin(MISSING_TOKENS))


def key_text(s: pd.Series) -> pd.Series:
    """Versiune pentru comparații: text curățat, litere mici, fără ® ™, cu diacriticele ş/ţ unificate."""
    return (clean_text(s).str.lower()
            .str.replace(r"[®™©]", "", regex=True)
            .str.replace("ş", "ș").str.replace("ţ", "ț")
            .str.replace(r"\s+", " ", regex=True).str.strip())


_DIV_RE = re.compile(r"n\s*(\d+)(?:\s*[x×]\s*(\d+))?", re.IGNORECASE)


def units_in_pack(divizare: str) -> float:
    """„N10x3” -> 30, „N28” -> 28. Formatele compuse („N1 + N1”) sau neclare -> NaN."""
    if not isinstance(divizare, str):
        return np.nan
    d = divizare.strip()
    if "+" in d:
        return np.nan
    m = _DIV_RE.match(d)
    if not m:
        return np.nan
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else 1
    return float(a * b)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        # pandas >= 3 folosește dtype-ul „str” pentru text; tratăm orice coloană non-numerică
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = clean_text(df[c])
    return df


def add_price_per_unit(cat: pd.DataFrame) -> pd.DataFrame:
    """Prețul efectiv în MDL = prețul negociat, dacă există, altfel prețul de producător.
    (În catalog fiecare produs are exact unul dintre cele două — verificat în EDA.)"""
    cat = cat.copy()
    cat["pret_mdl"] = cat["pret_negociat_mdl"].fillna(cat["pret_producator_mdl"])
    cat["tip_pret"] = np.where(cat["pret_negociat_mdl"].notna(), "negociat", "producator")
    cat["unitati"] = cat["divizare"].map(units_in_pack)
    cat["pret_unitate_mdl"] = cat["pret_mdl"] / cat["unitati"]
    return cat


def equivalence_key(df: pd.DataFrame) -> pd.Series:
    """Cheia grupului de echivalență de nivel 1 (propunerea de teză, secțiunea 3):
    DCI + doză + formă; pentru combinațiile fără DCI explicit („Combinaţie”) se folosește codul ATC complet.
    Aici doar pe text curățat — normalizarea dozelor/formelor vine în pasul următor."""
    dci = key_text(df["dci"])
    is_combo = dci.str.startswith("combina", na=False)
    base = dci.where(~is_combo, "atc:" + clean_text(df["atc"]).fillna("?"))
    return base + " | " + key_text(df["doza"]).fillna("?") + " | " + key_text(df["forma"]).fillna("?")
