"""Pasul 4 (validare) al pipeline-ului: setul semisintetic pentru antrenarea și ajustarea modelelor de potrivire.

Ideea: perechile Catalog ↔ Nomenclator au răspunsul corect cunoscut (același cod), dar textul lor este
identic în 99,9% din cazuri, deci sunt prea ușoare. Aici luăm rândurile Catalogului și le „murdărim”
controlat, ca să arate ca rândurile listei CNAM, apoi le trecem prin aceeași normalizare.
Răspunsul corect rămâne cel dat de cod.

Ce se introduce (pe textul brut, înainte de normalizare), cu probabilități estimate pe lista CNAM:
  - denumirea: greșeli de litere, terminații diferite („Amoxicilină”/„Amoxicilin”), un număr fără unitate
    lipit de nume („Emcor 2,5”), un cuvânt în plus („Clorură de sodiu - Jurabek”) sau lipsă;
  - forma: un cuvânt lipsă sau în plus („comprimate” / „comprimate filmate”), un sinonim apropiat
    (soluție / suspensie), rar o formă cu totul alta (greșeală a sursei);
  - doza: doar componenta principală a unei combinații, volumul omis („400 mg” / „400 mg/10 ml”),
    o zecimală rotunjită, doza lipsă;
  - firma: scrisă fără „(PROD: ...)” și fără țară, ca în jumătate din lista CNAM (formatul), sau, la
    mărcile unice, alt deținător (firma neactualizată în CNAM);
  - divizarea: aceeași cantitate scrisă altfel („N30” / „N10x3”), volumul mutat în divizare („100 ml N1”)
    sau lipsă.
Nu se schimbă numărul de unități din ambalaj: un alt ambalaj este alt produs, nu o variantă de scriere.
Rândurile fără pereche: pentru o parte din rânduri, produsul corect este scos din copia Nomenclatorului
(fie toată marca, fie doar acel produs, cu celelalte ambalaje păstrate).

Totul este determinist pentru un seed dat.
"""
import re

import numpy as np
import pandas as pd

from clean import key_text
from normalize import dci_stem, norm_denumire, normalize_frame, normalize_linkage, strip_accents

SEED = 42

# Ponderile tipurilor de variație în cadrul fiecărui câmp, din citirea diferențelor reale
# CNAM ↔ Nomenclator (notebook 03, secțiunea 2).
NAME_OPS = {"greseala": 0.35, "terminatie": 0.20, "numar": 0.15, "cuvant_in_plus": 0.15, "cuvant_lipsa": 0.15}
FORM_OPS = {"cuvant_lipsa": 0.40, "cuvant_in_plus": 0.20, "sinonim": 0.30, "alta_forma": 0.10}
DOSE_OPS = {"doar_principala": 0.30, "fara_volum": 0.30, "rotunjire": 0.15, "lipsa": 0.25}

FORM_SYNONYMS = [("solutie", "suspensie"), ("suspensie", "solutie"), ("orodispersabile", "dispersabile"),
                 ("dispersabile", "orodispersabile"), ("injectabila", "injectabila/perfuzabila"),
                 ("perfuzabila", "injectabila/perfuzabila")]
FORM_EXTRA = [("comprimate", "comprimate filmate"), ("pulbere de inhalat", "pulbere de inhalat unidoza"),
              ("capsule", "capsule moi"), ("injectabila", "injectabila/perfuzabila")]
BLISTERS = (10, 14, 7, 15, 20, 5, 6, 8, 12)   # mărimi obișnuite de blister
FORM_FUNCTION_WORDS = {"pentru", "si", "de", "cu", "in", "la"}


# ---------------------------------------------------------------------------------------------
# 1. Estimarea frecvențelor pe lista CNAM
# ---------------------------------------------------------------------------------------------
def probable_pairs(v: pd.DataFrame, min_name_sim: float = 0.85) -> pd.DataFrame:
    """Pentru fiecare rând, candidatul cel mai asemănător (scor simplu: suma asemănărilor),
    păstrat doar dacă denumirea e suficient de apropiată ca să fie, probabil, același produs."""
    score = (v.sim_denumire.fillna(0) + v.eq_doza.fillna(0.5) + v.eq_forma.fillna(0.5)
             + v.eq_unitati.fillna(0.5) + v.sim_firma.fillna(0.5))
    best = v.loc[score.groupby(v.ia).idxmax()]
    return best[best.sim_denumire >= min_name_sim]


def estimate_rates(best: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, firm_col: str) -> dict:
    """Cât de des diferă fiecare câmp între un rând CNAM și perechea lui probabilă din Nomenclator.
    Sunt limite superioare: o parte din perechile „probabile” sunt de fapt produse diferite."""
    A, B = a.loc[best.ia].reset_index(drop=True), b.loc[best.ib].reset_index(drop=True)
    best = best.reset_index(drop=True)
    div_text_differs = key_text(A.divizare).values != key_text(B.divizare).values
    return {
        "denumire": (best.niv_denumire < 3).mean(),
        "forma": (best.eq_forma == 0).mean(),
        "doza": ((best.eq_doza == 0) | A.doza_norm.isna()).mean(),
        "firma_alt_detinator": (best.sim_firma < 0.6).mean(),
        "firma_format_scurt": (~a[firm_col].fillna("").str.lower().str.contains(r"\(\s*prod")).mean(),
        "divizare_notatie": ((best.eq_unitati == 1) & div_text_differs).mean(),
        "volum_in_divizare": A.divizare.fillna("").str.contains(r"\d\s*ml", case=False).mean(),
        "volum_lipsa": (A.volum_norm.isna() & B.volum_norm.notna()).mean(),
    }


# ---------------------------------------------------------------------------------------------
# 2. Variațiile, câmp cu câmp (fiecare funcție primește textul și generatorul aleator)
# ---------------------------------------------------------------------------------------------
def _pick(ops: dict, rng, allowed=None):
    keys = [k for k in ops if allowed is None or k in allowed]
    p = np.array([ops[k] for k in keys])
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _typo(word: str, rng) -> str:
    """O greșeală de o literă: ștearsă, dublată, înlocuită sau inversată cu vecina (nu prima literă)."""
    letters = [i for i, ch in enumerate(word) if ch.isalpha() and i > 0]
    if not letters:
        return word
    i = letters[rng.integers(len(letters))]
    op = rng.choice(["sterge", "dubleaza", "inlocuieste", "inverseaza"])
    if op == "sterge":
        return word[:i] + word[i + 1:]
    if op == "dubleaza":
        return word[:i] + word[i] + word[i:]
    if op == "inlocuieste":
        return word[:i] + rng.choice(list("aeioulnrstcm")) + word[i + 1:]
    j = i + 1 if i + 1 < len(word) and word[i + 1].isalpha() else i - 1
    lo, hi = sorted((i, j))
    return word[:lo] + word[hi] + word[lo] + word[hi + 1:] if lo > 0 else word


def vary_name(name: str, dose: str, firm: str, rng):
    words = name.split()
    allowed = set(NAME_OPS)
    if len(words) < 2:
        allowed.discard("cuvant_lipsa")
    if not (isinstance(dose, str) and re.search(r"\d", dose)) or re.search(r"\d", name):
        allowed.discard("numar")                 # fără doză sau cu un număr deja în nume
    if not isinstance(firm, str):
        allowed.discard("cuvant_in_plus")
    op = _pick(NAME_OPS, rng, allowed)
    if op == "greseala":
        return _typo(name, rng), op
    if op == "terminatie":                       # „Amoxicilină” -> „Amoxicilin”, „Famotidin” -> „Famotidine”
        w = words[0].rstrip("®™")
        w = w[:-1] if w[-1:].lower() in ("ă", "a", "e") else w + rng.choice(["ă", "e"])
        return " ".join([w] + words[1:]), op
    if op == "numar":                            # „Emcor” -> „Emcor 2,5” (număr fără unitate)
        return f"{name} {re.search(r'\d+(?:[.,]\d+)?', dose).group(0)}", op
    if op == "cuvant_in_plus":                   # „Clorură de sodiu” -> „Clorură de sodiu - Jurabek”
        extra = re.findall(r"[A-Za-z]{3,}", firm)
        return (f"{name} {rng.choice(['- ', ''])}{extra[0].capitalize()}" if extra else _typo(name, rng)), op
    return " ".join(words[:-1]), op              # cuvânt lipsă: ultimul cuvânt


def vary_form(form: str, rng, other_forms: list):
    """other_forms: celelalte forme ale aceleiași substanțe în Nomenclator (confuzia reală din CNAM
    e între forme ale aceleiași substanțe, ex. comprimate filmate / capsule)."""
    t = strip_accents(pd.Series([form.lower()])).iloc[0]
    words = t.split()
    droppable = [i for i, w in enumerate(words) if i > 0 and w not in FORM_FUNCTION_WORDS]
    other_forms = [f for f in other_forms if f != strip_accents(key_text(pd.Series([form]))).iloc[0]]
    allowed = {"alta_forma"} if other_forms else set()
    if droppable:
        allowed.add("cuvant_lipsa")
    if any(a in t and b not in t for a, b in FORM_EXTRA):
        allowed.add("cuvant_in_plus")
    if any(re.search(rf"\b{a}\b", t) for a, _ in FORM_SYNONYMS):
        allowed.add("sinonim")
    if not allowed:
        return form, None
    op = _pick(FORM_OPS, rng, allowed)
    if op == "cuvant_lipsa":
        i = droppable[rng.integers(len(droppable))]
        return " ".join(words[:i] + words[i + 1:]), op
    if op == "cuvant_in_plus":
        a, b = [(a, b) for a, b in FORM_EXTRA if a in t and b not in t][0]
        return t.replace(a, b, 1), op
    if op == "sinonim":
        a, b = [(a, b) for a, b in FORM_SYNONYMS if re.search(rf"\b{a}\b", t)][0]
        return re.sub(rf"\b{a}\b", b, t, count=1), op
    return other_forms[rng.integers(len(other_forms))], op


def vary_dose(dose: str, rng):
    t = dose
    allowed = {"lipsa"}
    if re.search(r"\d\s*[a-zA-Zµ%]+\s*[+/]\s*\d", t):
        allowed.add("doar_principala")
    if re.search(r"/\s*\d*[.,]?\d*\s*ml\s*$", t, flags=re.I):
        allowed.add("fara_volum")
    if re.search(r"\d{2,}[.,]\d", t):
        allowed.add("rotunjire")
    op = _pick(DOSE_OPS, rng, allowed)
    if op == "doar_principala":                  # „10000 UI/7200 UI/400 UI” -> „10000 UI”
        return re.split(r"\s*[+/]\s*(?=\d)", t)[0], op
    if op == "fara_volum":                       # „400 mg/10 ml” -> „400 mg”
        return re.sub(r"\s*/\s*\d*[.,]?\d*\s*ml\s*$", "", t, flags=re.I), op
    if op == "rotunjire":                        # „28,5 mg” -> „28 mg”
        return re.sub(r"(\d{2,})[.,]\d+", r"\1", t, count=1), op
    return None, op


def short_firm(firm: str) -> str:
    """„Egis Pharmaceuticals PLC, Ungaria(PROD: ...)” -> „Egis Pharmaceuticals PLC” (formatul scurt din CNAM)."""
    main = re.split(r"\(\s*prod", firm, flags=re.I)[0]
    parts = [p.strip() for p in main.split(",")]
    return ", ".join(parts[:-1]) if len(parts) > 1 else parts[0]


def renotate_division(div: str, rng):
    """Aceeași cantitate, altă notație: „N10x3” -> „N30”, „N28” -> „N14x2”. None dacă nu se poate."""
    m = re.fullmatch(r"\s*N\s*(\d+)\s*(?:[x×]\s*(\d+))?\s*", div or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2) or 1)
    if b > 1:
        return f"N{a * b}"
    blisters = [s for s in BLISTERS if a % s == 0 and a // s > 1]   # „N30” -> „N10x3” (blister de 10, 3 blistere)
    if not blisters:
        return None
    s = blisters[rng.integers(len(blisters))]
    return f"N{s}x{a // s}"


# ---------------------------------------------------------------------------------------------
# 3. Construirea setului
# ---------------------------------------------------------------------------------------------
def is_generic(df: pd.DataFrame) -> pd.Series:
    """Denumire generică = începe cu numele substanței („Amoxicilin-BP”, „Famotidine Sopharma”).
    La acestea firma face diferența între produse, așa că nu le schimbăm deținătorul."""
    name = norm_denumire(df.denumire).fillna("").str.replace(r"[^a-z]", "", regex=True).str[:5]
    stem = dci_stem(df.dci).fillna("#").str[:5]
    return name == stem


def perturb(cat: pd.DataFrame, rates: dict, factor: float, rng, forms_by_dci: dict) -> pd.DataFrame:
    """Copia Catalogului cu variațiile introduse pe textul brut. Coloana „variatii” le enumeră.
    forms_by_dci: rădăcina DCI -> formele (text curățat) sub care apare substanța în Nomenclator."""
    out = cat.copy()
    generic = is_generic(cat)
    stems = dci_stem(cat.dci)
    p = {k: min(v * factor, 0.9) for k, v in rates.items()}
    log = [[] for _ in range(len(out))]
    for i, (idx, r) in enumerate(out.iterrows()):
        def put(col, val, tag):
            out.at[idx, col] = val
            log[i].append(tag)
        if isinstance(r.denumire, str) and rng.random() < p["denumire"]:
            val, op = vary_name(r.denumire, r.doza, r.detinator, rng)
            put("denumire", val, f"denumire:{op}")
        if isinstance(r.forma, str) and rng.random() < p["forma"]:
            val, op = vary_form(r.forma, rng, forms_by_dci.get(stems.loc[idx], []))
            if op:
                put("forma", val, f"forma:{op}")
        if isinstance(r.doza, str) and rng.random() < p["doza"]:
            val, op = vary_dose(r.doza, rng)
            put("doza", val, f"doza:{op}")
        if isinstance(r.detinator, str):
            firm = r.detinator
            if not generic.loc[idx] and rng.random() < p["firma_alt_detinator"]:
                # alt deținător: un producător din paranteză, dacă există, altfel o firmă oarecare din Catalog
                prods = re.findall(r"(?:PROD:|;\s*,)\s*([^,;]+)", firm)
                other = [x for x in prods if x.strip().lower() not in firm.split("(")[0].lower()]
                firm = other[0].strip() if other else cat.detinator.iloc[rng.integers(len(cat))]
                log[i].append("firma:alt_detinator")
            # formatul scurt e o proporție observată, nu o variație de amplificat
            if rng.random() < rates["firma_format_scurt"]:
                firm = short_firm(firm)
            out.at[idx, "detinator"] = firm
        if rng.random() < p["divizare_notatie"]:
            new = renotate_division(r.divizare, rng)
            if new:
                put("divizare", new, "divizare:notatie")
        if isinstance(r.volum, str):
            u = rng.random()
            if u < p["volum_lipsa"]:
                put("volum", None, "volum:lipsa")
            elif u < p["volum_lipsa"] + rates["volum_in_divizare"]:
                out.at[idx, "divizare"] = f"{r.volum} {out.at[idx, 'divizare']}"
                put("volum", None, "volum:in_divizare")
    out["variatii"] = [";".join(x) for x in log]
    return out


# Același produs, din punctul de vedere al potrivirii: Nomenclatorul dă uneori coduri diferite pentru
# aceeași cantitate ambalată diferit („N30” și „N10x3”), iar noi comparăm divizarea după numărul total de unități.
PRODUCT_KEY = ["denumire_norm", "doza_norm", "forma_norm", "unitati_norm", "volum_norm", "firma_norm"]


def product_class(nom: pd.DataFrame) -> pd.Series:
    """Clasa de produs a fiecărui rând din Nomenclator: rândurile identice pe PRODUCT_KEY au aceeași clasă."""
    return nom.groupby(PRODUCT_KEY, dropna=False).ngroup()


def build_semisynthetic(cat: pd.DataFrame, nom: pd.DataFrame, rates: dict, countries: set,
                        factor: float = 1.0, p_no_pair: float = 0.055, seed: int = SEED):
    """Construiește setul semisintetic.

    cat, nom: Catalogul și Nomenclatorul curățate (pasul 2), cu coloanele normalizate (pașii 3–4).
    rates: frecvențele estimate pe CNAM (estimate_rates); factor: de câte ori le amplificăm.
    p_no_pair: proporția rândurilor pentru care produsul corect este scos din Nomenclator.
    Returnează (a, b): a = rândurile „murdărite”, normalizate din nou, cu cod_adevarat și are_pereche;
    b = copia Nomenclatorului din care au fost scoase produsele rândurilor fără pereche.
    """
    rng = np.random.default_rng(seed)
    nom = nom.assign(clasa_produs=product_class(nom))
    code_class = nom.drop_duplicates("cod").set_index("cod").clasa_produs
    base = cat[cat.cod.isin(set(nom.cod))].copy()            # doar produsele cu răspuns cunoscut
    base = base.rename(columns={"cod": "cod_adevarat"}).reset_index(drop=True)

    # rândurile fără pereche: alternativ fără toată marca / fără acel produs (celelalte ambalaje rămân).
    # Scoaterea unei mărci lasă fără pereche și celelalte ambalaje ale ei din Catalog, așa că alegem
    # pe rând și ne oprim când proporția rândurilor fără pereche atinge p_no_pair.
    target = int(round(p_no_pair * len(base)))
    drop = np.zeros(len(nom), dtype=bool)
    mode = 0
    for j in rng.permutation(len(base)):
        n_lost = (~base.cod_adevarat.isin(set(nom.cod[~drop]))).sum()
        if n_lost >= target:
            break
        r = base.iloc[j]
        same_brand = (nom.denumire_norm == r.denumire_norm).values
        if mode == 0:
            new = same_brand
        else:
            new = (same_brand & (nom.doza_norm == r.doza_norm).values & (nom.forma_norm == r.forma_norm).values) \
                  | (nom.cod == r.cod_adevarat).values
        lost_if = (~base.cod_adevarat.isin(set(nom.cod[~(drop | new)]))).sum()
        if lost_if <= target * 1.02:                         # nu depășim ținta (toleranță 2%)
            drop |= new
            mode = 1 - mode
    b = nom[~drop].copy()

    raw_cols = ["cod_adevarat", "denumire", "forma", "doza", "volum", "divizare", "detinator", "tara", "dci", "atc"]
    forms = pd.DataFrame({"k": nom.dci_stem, "f": strip_accents(key_text(nom.forma))}).dropna()
    forms_by_dci = forms.groupby("k").f.agg(lambda x: sorted(set(x))).to_dict()
    a = perturb(base[raw_cols], rates, factor, rng, forms_by_dci)
    a = normalize_linkage(normalize_frame(a), ["detinator"], countries)   # aceeași normalizare ca pe CNAM
    a["clasa_adevarata"] = a.cod_adevarat.map(code_class)
    a["are_pereche"] = a.clasa_adevarata.isin(set(b.clasa_produs))
    a["fara_pereche_tip"] = np.where(a.are_pereche, "",
                                     np.where(a.denumire_norm.isin(set(b.denumire_norm)), "produs", "marca"))
    return a, b


def true_pairs(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Perechile corecte (ia, ib): produsul din b are aceeași clasă de produs ca produsul corect al rândului a
    (același cod sau alt cod pentru aceeași cantitate ambalată diferit)."""
    return (pd.DataFrame({"ia": a.index, "k": a.clasa_adevarata.values})
              .merge(pd.DataFrame({"ib": b.index, "k": b.clasa_produs.values}), on="k")[["ia", "ib"]])
