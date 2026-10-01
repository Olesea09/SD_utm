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


# ---------------------------------------------------------------------------------------------
# 5. Câmpuri suplimentare pentru potrivire: cheia DCI, firma, divizarea (adăugate în notebook 03)
# ---------------------------------------------------------------------------------------------
# Cuvinte care nu identifică substanța: sărurile și formele chimice („Metformini hydrochloridum”
# = „METFORMINUM”) și legăturile dintre componente.
DCI_STOP = {
    "acidum", "acidi", "natrii", "natrium", "natricum", "natrici", "dinatricum", "kalii", "kalium",
    "calcii", "calcium", "magnesii", "hydrochloridum", "hydrochloridi", "hydrobromidum", "sulfas",
    "propionas", "humanum", "dihydricum", "monohydricum", "trihydricum", "bromidum", "maleas",
    "mesilas", "besilas", "fumaras", "tartras", "citras", "phosphas", "acetas", "succinas",
    "plus", "si", "cu",
}


def _dci_stem_one(x):
    if not isinstance(x, str) or x.startswith("combina"):   # „Combinaţie” nu identifică substanța
        return None
    stems = set()
    for comp in re.split(r"\s*(?:\+|\bplus\b|,)\s*", x):
        words = [w for w in re.findall(r"[a-z]+", comp) if w not in DCI_STOP and len(w) > 2]
        if words:
            stems.add(words[0][:6])                         # rădăcina latină: primele 6 litere
    return "+".join(sorted(stems)) or None


def dci_stem(s: pd.Series) -> pd.Series:
    """Cheia DCI pe rădăcini: „Metformini hydrochloridum” și „METFORMINUM” -> „metfor”;
    „AMLODIPINUM+VALSARTANUM” și „Valsartanum + Amlodipinum” -> „amlodi+valsar”; „Combinaţie” -> NaN."""
    return strip_accents(key_text(s)).map(_dci_stem_one)


# Formele juridice și cuvintele generice din numele firmelor nu deosebesc firmele între ele.
FIRM_STOP = {
    "srl", "sa", "s", "a", "r", "l", "ltd", "limited", "pvt", "llc", "gmbh", "ag", "plc", "sc", "ics",
    "sap", "jsc", "co", "inc", "spa", "p", "kgaa", "as", "dd", "d", "o", "ks", "k", "kg", "bv", "nv",
    "ooo", "oao", "zao", "pao", "ao", "sl", "slu", "u", "sro", "doo", "se", "ab", "oy", "oyj", "corp",
    "corporation", "pharmaceutical", "pharmaceuticals", "pharma", "pharm", "farm", "laboratories",
    "laboratorios", "laboratorio", "industries", "industry", "and", "ve", "company", "int",
    "international", "sas", "sarl", "spol", "zo", "sp", "tic", "san",
}


def norm_firma(s: pd.Series, countries: set) -> pd.Series:
    """Numele firmei comparabil între surse: fără partea „(PROD: ...)”, fără țară, fără forma juridică.
    „Balkan Pharmaceuticals SRL, Republica Moldova(PROD: ...)” și „SC Balkan Pharmaceuticals SRL” -> „balkan”.
    `countries`: țările scrise ca în coloanele „tara”, trecute prin strip_accents(key_text(...))."""
    t = strip_accents(key_text(s)).str.replace(r"\(\s*prod\.?:.*$", "", regex=True)

    def one(x):
        if not isinstance(x, str):
            return None
        x = " ".join(p.strip() for p in x.split(",") if p.strip() not in countries)
        words = [w for w in re.findall(r"[a-z0-9]+", x) if w not in FIRM_STOP and len(w) > 1]
        return " ".join(words) or None
    return t.map(one)


def firm_set(df: pd.DataFrame, cols: list, countries: set) -> pd.Series:
    """Toate firmele asociate unui rând (deținător și producători), normalizate, separate prin „|”.
    Formatele acceptate: „Firma A, Țara(PROD: Firma B, Țara; ...)”, „Firma A (prod.: Firma B, Țara)”
    și lista Nomenclatorului „Firma A, Țara; Firma B, Țara”. Fiecare bucată trece prin norm_firma."""
    def pieces(x):
        if not isinstance(x, str):
            return []
        x = strip_accents(pd.Series([x.lower()])).iloc[0]
        main, _, prod = x.partition("(prod")
        prod = re.sub(r"^\.?:", "", prod).replace(")", " ").replace("(", " ")
        return [p for p in re.split(r";", main) + re.split(r";", prod) if p.strip(" ,")]

    out = []
    for _, row in df[cols].iterrows():
        ps = [p for c in cols for p in pieces(row[c])]
        names = norm_firma(pd.Series(ps, dtype="object"), countries).dropna() if ps else []
        out.append("|".join(dict.fromkeys(names)) or None)
    return pd.Series(out, index=df.index)


_VOL = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|l|g|doze)\b")
_N = re.compile(r"n\s*(\d+)(?:\s*x\s*(\d+))?")


def _volume(x):
    """„100 ml” -> „100 ml”; „1 l” -> „1000 ml”; „20g” -> „20 ml” (g = ml doar pentru potrivire); „60 doze”."""
    if not isinstance(x, str):
        return None
    m = _VOL.search(x)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    v, u = (v * 1000, "ml") if u == "l" else (v, "doze" if u == "doze" else "ml")
    return f"{_fmt(v)} {u}"


def norm_divizare(div: pd.Series, vol: pd.Series) -> pd.DataFrame:
    """Divizarea ca număr total de unități + volumul, separat.
    „N14x2” și „N28” -> 28; „60 ml N1” -> 1 unitate și volumul 60 ml; „N1 + N1” (trusă) -> 1;
    volumul se ia din coloana „volum”, iar dacă lipsește, din textul divizării."""
    t = strip_accents(key_text(div)).str.replace(",", ".", regex=False)
    t_no_par = t.str.replace(r"\([^)]*\)", " ", regex=True)            # „N1(flacon PP)” -> „N1”

    def units(x):
        if not isinstance(x, str):
            return None
        m = _N.search(x)                                               # primul grup N (la truse: prima componentă)
        return float(int(m.group(1)) * (int(m.group(2)) if m.group(2) else 1)) if m else None
    v = strip_accents(key_text(vol)).str.replace(",", ".", regex=False).map(_volume)
    return pd.DataFrame({"unitati_norm": t_no_par.map(units),
                         "volum_norm": v.where(v.notna(), t_no_par.map(_volume))}, index=div.index)


def normalize_linkage(df: pd.DataFrame, firm_cols: list, countries: set) -> pd.DataFrame:
    """Adaugă coloanele folosite la blocare și comparare: dci_stem, firma_norm (deținătorul),
    firme (deținător + producători), unitati_norm, volum_norm. Coloanele originale rămân neschimbate."""
    df = df.copy()
    df["dci_stem"] = dci_stem(df["dci"])
    df["firma_norm"] = norm_firma(df[firm_cols[0]], countries)
    df["firme"] = firm_set(df, firm_cols, countries)
    df[["unitati_norm", "volum_norm"]] = norm_divizare(df["divizare"], df["volum"])
    return df


def country_set(*cols: pd.Series) -> set:
    """Țările care apar în coloanele „tara” ale surselor, în forma folosită de norm_firma."""
    return set(strip_accents(key_text(pd.concat(cols))).dropna())
