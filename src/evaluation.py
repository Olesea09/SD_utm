"""Pasul 8 al pipeline-ului: evaluarea finală pe rândurile CNAM etichetate manual.

Eșantionul etichetat este stratificat: 100 de rânduri din cele potrivite exact după normalizare și 100 din cele
nepotrivite. În lista completă straturile au mărimi foarte diferite (1.972 și 195), așa că fiecare rând
primește o pondere = mărimea stratului / mărimea eșantionului din strat (19,72 și 1,95), iar metricile
sunt medii ponderate. Astfel ele estimează performanța pe întreaga listă CNAM, nu doar pe eșantion.

Răspunsul corect se compară la nivelul clasei de produs (aceeași regulă ca pe setul semisintetic):
un cod ales de model este corect dacă are aceeași clasă de produs ca un cod ales la etichetare.
"""
import numpy as np
import pandas as pd

SEED = 42


def label_truth(labels: pd.DataFrame, nom: pd.DataFrame) -> pd.DataFrame:
    """Din etichete: tipul răspunsului („pereche”, „niciunul”, „incert”) și clasele de produs corecte.
    labels: rezultatul labeling.read_labels (index = id_cnam); nom: Nomenclatorul cu coloanele cod și clasa_produs."""
    classes_of = nom.groupby("cod").clasa_produs.agg(lambda s: set(s))    # un cod poate apărea de mai multe ori
    out = pd.DataFrame(index=labels.index)
    out["eticheta"] = np.where(labels.status.eq("INCERT"), "incert",
                               np.where(labels.cod_corect.notna(), "pereche", "niciunul"))
    out["cod_corect"] = labels.cod_corect
    out["clase_corecte"] = [classes_of.get(c, set()) if isinstance(c, str) else set() for c in labels.cod_corect]
    return out


def outcomes(decision_code: pd.Series, truth: pd.DataFrame, nom: pd.DataFrame) -> pd.DataFrame:
    """Rezultatul fiecărui rând etichetat pentru un model. decision_code: codul ales (NaN = fără pereche), index id_cnam.
    Coloane: legat, corect (legătura e corectă), are_pereche, bine (rândul e tratat corect, inclusiv „fără pereche”)."""
    cls = nom.groupby("cod").clasa_produs.agg(lambda s: set(s))
    t = truth.copy()
    code = decision_code.reindex(t.index)
    t["legat"] = code.notna()
    t["corect"] = [bool(cls.get(c, set()) & ok) if isinstance(c, str) else False for c, ok in zip(code, t.clase_corecte)]
    t["are_pereche"] = t.eticheta.eq("pereche")
    t["bine"] = np.where(t.are_pereche, t.corect, ~t.legat)
    return t


def weighted_metrics(o: pd.DataFrame, w: pd.Series) -> dict:
    """Precizie, recall, F1 și acuratețe ponderate (rândurile „incert” trebuie excluse înainte)."""
    w = w.reindex(o.index).to_numpy(float)
    link, corr, has, ok = (o[c].to_numpy(bool) for c in ("legat", "corect", "are_pereche", "bine"))
    prec = (w * (link & corr)).sum() / (w * link).sum()
    rec = (w * (link & corr)).sum() / (w * has).sum()
    return {"precizie": prec, "recall": rec, "F1": 2 * prec * rec / (prec + rec),
            "acuratețe": (w * ok).sum() / w.sum(),
            "legături greșite": int((link & ~corr).sum()), "perechi ratate": int((has & ~link).sum()),
            "„fără pereche” legate": int((~has & link).sum())}


def stratified_bootstrap(o: pd.DataFrame, w: pd.Series, strata: pd.Series, n_boot: int = 2000, seed: int = SEED):
    """Intervale de încredere 95% prin bootstrap stratificat: rândurile se reeșantionează în interiorul fiecărui strat,
    ca proporțiile straturilor și ponderile să rămână cele ale eșantionului."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata.reindex(o.index).to_numpy() == s) for s in strata.unique()]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups if len(g)])
        m = weighted_metrics(o.iloc[idx].reset_index(drop=True), w.reindex(o.index).iloc[idx].reset_index(drop=True))
        vals.append([m["precizie"], m["recall"], m["F1"], m["acuratețe"]])
    lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
    return pd.DataFrame({"inf": lo, "sup": hi}, index=["precizie", "recall", "F1", "acuratețe"])
