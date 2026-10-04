"""Pasul 9 al pipeline-ului: analiza prețurilor de producător pe datele integrate.

Toate prețurile sunt prețuri de producător (ex works) din Catalogul național de prețuri: prețul negociat dacă
există, altfel prețul de producător. Ele NU sunt prețurile plătite în farmacie. Se analizează doar formele
solide, la care prețul pe unitate (comprimat, capsulă) este comparabil între ambalaje. Prețurile sunt
modelate pe scară logaritmică, pentru că variază pe cinci ordine de mărime.

Grup de echivalență = DCI + doză + formă normalizate (codul ATC complet la combinațiile fără DCI).
Un produs este „compensat” dacă codul lui a fost legat de un rând CNAM de modelul ales în notebook 05.
"""
import numpy as np
import pandas as pd

from clean import key_text
from normalize import equivalence_key_norm, norm_firma

SEED = 42
SOLID = r"comprimat|capsul|drajeu|supozitor|ovul|plic"


def price_table(cat: pd.DataFrame, linked: pd.DataFrame, countries: set) -> pd.DataFrame:
    """Produsele solide din Catalog cu preț pe unitate, grupul de echivalență, deținătorul și legătura cu CNAM.
    linked: lista CNAM legată (notebook 05), cu cod_nomenclator, probabilitate și de_verificat."""
    d = cat.copy()
    d["grup"] = equivalence_key_norm(d)
    d["firma"] = norm_firma(d.detinator, countries)
    d = d[key_text(d.forma).str.contains(SOLID, na=False) & (d.pret_unitate_mdl > 0)].copy()
    d["y"] = np.log(d.pret_unitate_mdl)                                   # log(preț pe unitate, MDL)
    lk = (linked.dropna(subset=["cod_nomenclator"])
                .groupby("cod_nomenclator").agg(prob_legatura=("probabilitate", "max"),
                                                de_verificat=("de_verificat", "all")))
    d["compensat"] = d.cod.isin(lk.index)
    d = d.join(lk, on="cod")
    return d.reset_index(drop=True)


def group_stats(d: pd.DataFrame) -> pd.DataFrame:
    """Pentru fiecare grup cu cel puțin două produse: numărul de produse, de deținători, raportul max/min
    al prețului pe unitate și abaterea standard a log(preț)."""
    g = d.groupby("grup").agg(n=("y", "size"), detinatori=("firma", "nunique"),
                              compensate=("compensat", "sum"), y_min=("y", "min"), y_max=("y", "max"),
                              sd=("y", "std"))
    g = g[g.n >= 2].copy()
    g["raport"] = np.exp(g.y_max - g.y_min)
    return g


def mechanical_null(g: pd.DataFrame, sigma: float, n_sim: int = 1000, seed: int = SEED):
    """Ipoteza nulă „fără efect real”: toate grupurile au aceeași dispersie sigma, deci raportul max/min depinde
    doar de numărul de produse n. Simulăm log-prețurile N(0, sigma) pentru fiecare grup și întoarcem
    raportul max/min simulat (n_sim × grupuri)."""
    rng = np.random.default_rng(seed)
    n = g.n.to_numpy()
    out = np.empty((n_sim, len(n)))
    for k in range(n_sim):
        out[k] = [np.exp(np.ptp(rng.normal(0, sigma, m))) for m in n]
    return out


def pooled_sigma(d: pd.DataFrame, groups) -> float:
    """Abaterea standard comună în interiorul grupurilor (estimatorul combinat)."""
    x = d[d.grup.isin(groups)]
    r = x.y - x.groupby("grup").y.transform("mean")
    return float(np.sqrt((r ** 2).sum() / (len(x) - x.grup.nunique())))


# ---------------------------------------------------------------------------------------------
# Efectul „compensat” în același grup: frecventist (efecte fixe) și bayesian (model ierarhic)
# ---------------------------------------------------------------------------------------------
def mixed_groups(d: pd.DataFrame) -> pd.DataFrame:
    """Produsele din grupurile care conțin și produse compensate, și necompensate."""
    g = d.groupby("grup").compensat.agg(["sum", "size"])
    keep = g[(g["sum"] >= 1) & (g["sum"] < g["size"])].index
    return d[d.grup.isin(keep)].copy()


def _fe_parts(m: pd.DataFrame) -> pd.DataFrame:
    """Contribuția fiecărui grup la estimatorul cu efecte fixe: numărătorul sum(cd * yd) și numitorul sum(cd^2),
    unde yd și cd sunt abaterile lui log(preț) și ale indicatorului „compensat” de la media grupului."""
    yd = m.y - m.groupby("grup").y.transform("mean")
    cd = m.compensat.astype(float) - m.groupby("grup").compensat.transform("mean")
    return pd.DataFrame({"num": cd * yd, "den": cd ** 2, "grup": m.grup}).groupby("grup").sum()


def fixed_effects_delta(m: pd.DataFrame) -> float:
    """Estimatorul cu efecte fixe de grup: diferența medie de log(preț) dintre produsele compensate și cele
    necompensate din același grup (regresia pe abaterile de la media grupului)."""
    p = _fe_parts(m)
    return float(p.num.sum() / p.den.sum())


def cluster_bootstrap_fe(m: pd.DataFrame, n_boot: int = 2000, seed: int = SEED) -> np.ndarray:
    """Bootstrap pe grupuri pentru estimatorul cu efecte fixe: se reeșantionează grupuri întregi, pentru că
    produsele aceluiași grup nu sunt independente. Estimatorul este o sumă pe grupuri, deci e suficient
    să reeșantionăm contribuțiile grupurilor."""
    rng = np.random.default_rng(seed)
    p = _fe_parts(m)
    num, den = p.num.to_numpy(), p.den.to_numpy()
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    return num[idx].sum(1) / den[idx].sum(1)


def bayes_compensated(m: pd.DataFrame, draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = SEED):
    """Model bayesian ierarhic:  y_ij = alpha_j + delta * compensat_ij + e_ij,  e_ij ~ N(0, sigma),
    alpha_j ~ N(mu, tau) (nivelul de preț al grupului j, „împrumută” informație de la celelalte grupuri).
    A priori slab informative: mu ~ N(media lui y, 5), tau, sigma ~ HalfNormal(2), delta ~ N(0, 1).
    delta este diferența de log(preț) compensat – necompensat; exp(delta) - 1 = diferența procentuală.
    Notă de reproductibilitate: cu același seed, eșantioanele pot diferi foarte puțin între rulări (erori de
    rotunjire de ordinul 1e-9 în gradient, amplificate de NUTS); diferențele rămân sub eroarea Monte Carlo."""
    import pymc as pm
    codes, groups = pd.factorize(m.grup)
    with pm.Model() as model:
        mu = pm.Normal("mu", float(m.y.mean()), 5)
        tau = pm.HalfNormal("tau", 2)
        z = pm.Normal("z", 0, 1, shape=len(groups))                    # parametrizare necentrată
        alpha = pm.Deterministic("alpha", mu + tau * z)
        delta = pm.Normal("delta", 0, 1)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y", alpha[codes] + delta * m.compensat.to_numpy(float), sigma, observed=m.y.to_numpy())
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                          progressbar=False, target_accept=0.95)
    return model, idata


# ---------------------------------------------------------------------------------------------
# Dispersia și numărul de deținători: model bayesian ierarhic cu dispersie proprie fiecărui grup
# ---------------------------------------------------------------------------------------------
FLOOR = 0.01   # diferențele de preț sub ~1% sunt tratate ca neglijabile (rotunjiri)


def bayes_dispersion(d: pd.DataFrame, g: pd.DataFrame, draws: int = 1000, tune: int = 1000, chains: int = 4,
                     seed: int = SEED, floor: float = FLOOR):
    """log sigma_j = g0 + g1 * log(deținători_j) + s * eta_j,  eta_j ~ N(0, 1).
    sigma_j este dispersia prețurilor în grupul j (abaterea standard a log-prețului). Spre deosebire de raportul
    max/min, ea nu crește mecanic cu numărul de produse, deci g1 > 0 înseamnă un efect real: grupurile cu mai
    mulți deținători au prețuri mai împrăștiate. Grupurile mici își estimează sigma_j „împrumutând” informație
    de la celelalte, prin g0, g1 și s.
    Media fiecărui grup este integrată analitic (a priori uniformă): toată informația despre sigma_j se află în
    suma pătratelor abaterilor de la media grupului, SS_j, iar SS_j / sigma_j^2 urmează o distribuție
    chi-pătrat cu n_j - 1 grade de libertate. Log-verosimilitatea grupului j este deci
        -(n_j - 1) log sigma_j - SS_j / (2 sigma_j^2)   (plus o constantă).
    Aproximativ 18% dintre grupuri au prețuri practic identice (SS_j aproape 0), iar pentru ele sigma_j ar coborî
    spre 0 și log sigma_j spre minus infinit. Folosim de aceea o dispersie efectivă sigma_j^2 + floor^2, cu floor = 0,01:
    sub o diferență de aproximativ 1% datele nu mai pot deosebi dispersiile între ele."""
    import pymc as pm
    x = d[d.grup.isin(g.index)]
    ss = x.groupby("grup").y.agg(lambda v: float(((v - v.mean()) ** 2).sum())).reindex(g.index).to_numpy()
    df = (g.n - 1).to_numpy(float)
    log_h = np.log(g.detinatori.to_numpy(float))
    with pm.Model() as model:
        g0 = pm.Normal("g0", -2, 1)
        g1 = pm.Normal("g1", 0, 1)
        s = pm.HalfNormal("s", 1)
        eta = pm.Normal("eta", 0, 1, shape=len(g))                     # parametrizare necentrată
        log_sigma = pm.Deterministic("log_sigma", g0 + g1 * log_h + s * eta)
        var = pm.math.exp(2 * log_sigma) + floor ** 2                  # dispersia efectivă, cu pragul de 1%
        pm.Potential("verosimilitate", (-df / 2 * pm.math.log(var) - ss / (2 * var)).sum())
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                          progressbar=False, target_accept=0.95)
    return model, idata
