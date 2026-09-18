"""Pasul 1 al pipeline-ului: citirea surselor brute (data/raw) într-un format tabelar uniform.

Fișierele din data/raw NU se modifică niciodată. Toate funcțiile returnează DataFrame-uri
cu nume de coloane scurte, standardizate, pentru a putea fi comparate între surse.
Toate câmpurile sunt citite ca text (dtype=str), pentru a nu pierde zerourile din fața
codurilor (ex. „0500090185”); conversiile numerice se fac explicit.
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

NOMENCLATOR_COLS = {
    "Nr. d/o": "nr", "Codul medicamentului": "cod", "Denumirea comercială": "denumire",
    "Forma farmaceutică": "forma", "Doza, concentraţia": "doza", "Volum": "volum",
    "Divizarea": "divizare", "Deținător al CÎM": "detinator", "Ţara": "tara",
    "Firma producătoare": "producator", "Numărul de înregistrare": "nr_inregistrare",
    "Data înregistrării": "data_inregistrare", "Codul ATC": "atc",
    "Denumirea comună internaţională": "dci", "Termenul de valabilitate": "valabilitate_luni",
    "Codul cu bare": "cod_bare", "Tip cerere": "tip_cerere", "Data expirării CÎM": "data_expirare_cim",
    "Presciptie": "prescriptie", "Tip medicament": "tip_medicament",
}

CATALOG_COLS = [
    "cod", "denumire", "forma", "doza", "volum", "divizare", "tara", "detinator",
    "nr_inregistrare", "data_inregistrare", "atc", "dci", "valabilitate_luni", "cod_bare",
    "pret_producator_mdl", "pret_producator_valuta", "pret_negociat_mdl", "pret_negociat_valuta",
    "valuta", "aprobare_pret", "modificari", "valabilitate_pret_negociat",
]

CNAM_ADULTI = [
    "dci", "denumire", "doza", "forma", "divizare", "volum", "maladie", "durata_tratament",
    "statut_compensare", "suma_fixa_unitate", "suma_fixa_cutie", "tara", "producator",
]
CNAM_COPII = [c for c in CNAM_ADULTI if c != "suma_fixa_cutie"]
# ATENȚIE (constatare EDA): în fișierul RU antetele coloanelor 4 și 5 sunt inversate
# („Упаковка” stă deasupra formei farmaceutice, „Фармацевтическая форма” deasupra divizării).
# Coloanele sunt numite după CONȚINUTUL lor real, nu după antet, deci aceeași listă ca la RO.

PRICE_COLS = ["pret_producator_mdl", "pret_producator_valuta", "pret_negociat_mdl", "pret_negociat_valuta"]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def load_nomenclator() -> pd.DataFrame:
    df = pd.read_excel(RAW / "nomenclator_amdm.xls", sheet_name=0, dtype=str, engine="xlrd")
    df = df.iloc[1:]  # al doilea rând conține doar numerotarea coloanelor (0, 1, 2, ...)
    df = df.rename(columns=NOMENCLATOR_COLS)
    return df.reset_index(drop=True)


def load_catalog() -> pd.DataFrame:
    df = pd.read_excel(RAW / "catalog_preturi_amdm.xls", sheet_name=0, dtype=str, engine="xlrd")
    df.columns = CATALOG_COLS
    for c in PRICE_COLS:
        df[c] = _to_num(df[c])
    return df


def _load_cnam(fname: str, lang: str) -> pd.DataFrame:
    xl = pd.ExcelFile(RAW / fname)
    parts = []
    for sheet, grup in zip(xl.sheet_names, ["adulti", "copii"]):
        d = pd.read_excel(xl, sheet_name=sheet, dtype=str)
        d.columns = CNAM_ADULTI if grup == "adulti" else CNAM_COPII
        d["grup"] = grup
        d["limba"] = lang
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    for c in ["suma_fixa_unitate", "suma_fixa_cutie"]:
        df[c] = _to_num(df[c])
    return df


def load_cnam_ro() -> pd.DataFrame:
    return _load_cnam("cnam_compensate_RO.xlsx", "ro")


def load_cnam_ru() -> pd.DataFrame:
    return _load_cnam("cnam_compensate_RU.xlsx", "ru")


def load_all() -> dict:
    return {"nomenclator": load_nomenclator(), "catalog": load_catalog(),
            "cnam_ro": load_cnam_ro(), "cnam_ru": load_cnam_ru()}
