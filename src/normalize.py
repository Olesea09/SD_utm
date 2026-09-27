"""Pasul 3 al pipeline-ului: normalizarea semantică a denumirii, dozei și formei farmaceutice.

Ce face:  aduce aceeași informație, scrisă diferit în surse diferite, la o formă canonică unică:
          - forma: abrevierile CNAM („comp. film.”) -> denumirea completă („comprimate filmate”),
            „pentru” / „și” / spațiile din jurul lui „/” scrise uniform, fără diacritice;
          - doza: „+” -> „/” la combinații, unități standard (g, mcg -> mg; ui, ua, u, unități -> ui),
            concentrațiile aduse la „pe ml” („50 mg/2 ml” -> „25 mg/ml”), volumul ambalajului eliminat;
          - denumirea: doza și forma lipite de nume eliminate („ciprinol 500 mg” -> „ciprinol”).
Aceleași funcții se aplică IDENTIC tuturor surselor (și Nomenclatorul are inconsecvențe proprii).
Coloanele originale rămân neschimbate; se adaugă coloane noi cu sufixul „_norm”.
Ce NU face: nu corectează greșelile de scriere („bicilin” vs „bicillin”); acelea sunt tratate
          de potrivirea probabilistă (pasul următor), care cântărește similaritatea.
"""
import re
import unicodedata
import pandas as pd

from clean import key_text


def strip_accents(s: pd.Series) -> pd.Series:
    """„soluție orală” -> „solutie orala” (CNAM scrie uneori „orala”, fără diacritice)."""
    return s.map(lambda x: "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))
                 if isinstance(x, str) else x)


# ---------------------------------------------------------------------------------------------
# 1. Forma farmaceutică
# ---------------------------------------------------------------------------------------------
# Dicționarul de abrevieri, construit din cele 43 de forme CNAM care nu apar în Nomenclator
# (notebook 02, secțiunea 1). Se aplică pe text fără diacritice, în ordinea de mai jos.
FORM_ABBREV = [
    (r"\bcomp\.", "comprimate "), (r"\bcaps\.", "capsule "), (r"\bpulb\.", "pulbere "),
    (r"\bsolv\.", "solvent "), (r"\bsol\.", "solutie "), (r"\bsusp\.", "suspensie "),
    (r"\binj\.", "injectabila "), (r"\bperf\.", "perfuzabila "), (r"\bconc\.", "concentrat "),
    (r"\bgran\.", "granule "), (r"\bsup\.", "supozitoare "), (r"\bpic\.", "picaturi "),
    (r"\bfilm\.", "filmate "), (r"\beferv\.", "efervescente "), (r"\bgastrorez\.", "gastrorezistente "),
    (r"\bmasticab\.", "masticabile "), (r"\belib\.", "cu eliberare "), (r"\bmodif\.", "modificata "),
    (r"\bprel\.", "prelungita "),
]
# Recipientul: „, cartuș” (CNAM) = „în cartuș” (Nomenclator). Recipientele pe care Nomenclatorul
# nu le folosește deloc în formă (flacon, pungă) și „în plic” se elimină. Seringa și stiloul
# preumplut rămân, pentru că Nomenclatorul le tratează ca forme distincte.
CONTAINER_RULES = [
    (r",\s*cartus\b", " in cartus"),
    (r"\s*\((flacon|punga)\)|,\s*(flacon|punga)\b", ""),
    (r"\s+in\s+plic\b", ""),
]


def norm_forma(s: pd.Series) -> pd.Series:
    t = strip_accents(key_text(s))
    for pat, rep in CONTAINER_RULES + FORM_ABBREV:
        t = t.str.replace(pat, rep, regex=True)
    return (t.str.replace(r"\s+pentru\s+", "/", regex=True)   # „pulbere pentru soluție” = „pulb./sol.”
             .str.replace(r"\s+si\s+", "+", regex=True)        # „pulbere și solvent” = „pulb.+solv.”
             .str.replace(",", " ", regex=False)
             .str.replace(r"\s*([/+])\s*", r"\1", regex=True)  # „ / ” și „/” scrise la fel
             .str.replace(r"\s+", " ", regex=True).str.strip())


# ---------------------------------------------------------------------------------------------
# 2. Doza
# ---------------------------------------------------------------------------------------------
# unitate -> (unitatea standard, factorul de conversie)
UNITS = {
    "mg": ("mg", 1), "g": ("mg", 1000), "mcg": ("mg", 1e-3), "µg": ("mg", 1e-3), "μg": ("mg", 1e-3),
    "micrograme": ("mg", 1e-3),
    "ui": ("ui", 1), "u": ("ui", 1), "ua": ("ui", 1), "iu": ("ui", 1), "unitati": ("ui", 1),
    "mbq": ("mbq", 1), "gbq": ("mbq", 1000), "mmol": ("mmol", 1),
}
VOLUME_UNITS = {"ml": 1, "l": 1000}
_TOKEN = re.compile(r"^(\d+(?:\.\d+)?)?\s*([a-zµμ%]+)$")


def _fmt(x: float) -> str:
    """4 cifre semnificative, fără zerouri inutile: 1500.0 -> „1500”, 0.040 -> „0.04”."""
    v = float(f"{x:.4g}")
    return f"{v:f}".rstrip("0").rstrip(".")


def parse_doza(text):
    """Doza ca text canonic, sau None dacă nu poate fi interpretată.

    „875 mg + 125 mg” -> „125 mg/875 mg”;  „1 g” -> „1000 mg”;  „50 mg/2 ml” -> „25 mg/ml”;
    „100 ui/ml - 3 ml” -> „100 ui/ml”;  „0,1%” -> „1 mg/ml”;  „50 mcg/250 mcg/doză” -> „0.05 mg/0.25 mg”.
    Componentele unei combinații se ordonează, pentru că ordinea diferă între surse („5 mg/160 mg”).
    """
    if not isinstance(text, str):
        return None
    t = strip_accents(pd.Series([text.lower()])).iloc[0]
    t = t.replace(",", ".").replace("+", "/")
    t = re.sub(r"\s*-\s*\d+(\.\d+)?\s*ml\s*$", "", t)         # volumul ambalajului: „- 3 ml”
    t = re.sub(r"/\s*doz[ae]\s*$", "", t)                      # „/doză” (inhalatoare)
    tokens = [x.strip() for x in t.split("/") if x.strip()]
    parsed = []
    for tok in tokens:
        m = _TOKEN.match(tok.replace(" ", ""))
        if not m:
            return None
        parsed.append((float(m.group(1)) if m.group(1) else None, m.group(2)))
    if not parsed:
        return None

    # numitorul concentrației: ultimul element, dacă e volum („5 ml”, „ml”) sau „g” fără număr („mg/g”)
    denom = 1.0
    num, unit = parsed[-1]
    if len(parsed) > 1 and (unit in VOLUME_UNITS or (unit == "g" and num is None)):
        denom = (num or 1) * VOLUME_UNITS.get(unit, 1)
        parsed, per = parsed[:-1], "/ml"   # g și ml tratate la fel (densitate ≈ 1), doar pentru potrivire
    else:
        per = ""

    parts = []
    for num, unit in parsed:
        if unit == "%":                     # 1% = 10 mg/ml (sau mg/g)
            if num is None or per:
                return None
            parts.append(("mg", num * 10)); per = "/ml"
            continue
        if num is None or unit not in UNITS:
            return None
        std, f = UNITS[unit]
        parts.append((std, num * f / denom))
    return "/".join(f"{_fmt(v)} {u}" for u, v in sorted(parts)) + per


def norm_doza(s: pd.Series) -> pd.Series:
    """Doza canonică; dacă nu poate fi interpretată, textul curățat cu „+” -> „/” (ca să nu pierdem rândul)."""
    fallback = strip_accents(key_text(s)).str.replace(r"\s*\+\s*", "/", regex=True)
    parsed = s.map(parse_doza)
    return parsed.where(parsed.notna(), fallback)


# ---------------------------------------------------------------------------------------------
# 3. Denumirea comercială
# ---------------------------------------------------------------------------------------------
_DOSE_IN_NAME = re.compile(r"\s*\d+(?:[.,]\d+)?\s*(mg|g|mcg|µg|ui|ml|%)\b.*$")


def norm_denumire(s: pd.Series) -> pd.Series:
    """„Lisinopril ATB 10 mg comprimate” -> „lisinopril atb”; „Metronidazol- BP” -> „metronidazol-bp”."""
    t = strip_accents(key_text(s))
    t = t.str.replace(r"\s*-\s*", "-", regex=True)             # spații în jurul cratimei
    t = t.str.replace(r"([a-z])(\d)", r"\1 \2", regex=True)    # „zentiva100 mg” -> „zentiva 100 mg”
    t = t.str.replace(_DOSE_IN_NAME, "", regex=True)           # doza (și ce urmează după ea)
    return t.str.strip()


# ---------------------------------------------------------------------------------------------
# 4. Aplicarea pe un tabel
# ---------------------------------------------------------------------------------------------
def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adaugă denumire_norm, doza_norm, forma_norm; coloanele originale rămân neschimbate."""
    df = df.copy()
    df["denumire_norm"] = norm_denumire(df["denumire"])
    df["doza_norm"] = norm_doza(df["doza"])
    df["forma_norm"] = norm_forma(df["forma"])
    return df


def equivalence_key_norm(df: pd.DataFrame) -> pd.Series:
    """Cheia grupului de echivalență de nivel 1 pe date normalizate: DCI + doză + formă
    (codul ATC complet pentru combinațiile fără DCI explicit), ca în clean.equivalence_key."""
    dci = key_text(df["dci"])
    is_combo = dci.str.startswith("combina", na=False)
    base = dci.where(~is_combo, "atc:" + key_text(df["atc"]).fillna("?"))
    return base + " | " + df["doza_norm"].fillna("?") + " | " + df["forma_norm"].fillna("?")
