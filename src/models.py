"""Pasul 7b al pipeline-ului: modelele supervizate și evaluarea potrivirii.

Modelele supervizate învață din perechi cu răspuns cunoscut (setul semisintetic):
  - regresia logistică: o sumă ponderată a scorurilor de asemănare, trecută printr-o funcție sigmoidă;
  - SVM (mașină cu vectori suport, nucleu RBF): granița cu cea mai mare margine dintre perechile corecte și
    cele greșite; probabilitățile se obțin prin calibrare Platt;
  - rețeaua neuronală (perceptron multistrat): poate învăța combinații de câmpuri.
Toate primesc aceleași caracteristici: scorurile continue din vectorul de comparare, cu valorile lipsă
înlocuite cu 0 și câte un indicator „lipsă” pentru câmpurile care pot lipsi.

Evaluarea se face pe două niveluri:
  - pe perechi: aria sub curba precizie–recall (AP) și scorul Brier (cât de bine calibrate sunt probabilitățile);
  - pe rânduri (decizia care contează în practică): pentru fiecare rând se alege candidatul cu probabilitatea
    cea mai mare, iar dacă ea depășește pragul, rândul primește codul lui; altfel „fără pereche”.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from compare import SIM_COLS

SEED = 42
# câmpurile care pot lipsi în una dintre surse; pentru fiecare se adaugă un indicator 0/1
MISSING_FLAGS = {"lipsa_doza": "eq_doza", "lipsa_volum": "eq_volum", "lipsa_firma": "sim_firma",
                 "lipsa_unitati": "eq_unitati", "lipsa_dci": "eq_dci", "lipsa_denumire": "sim_denumire"}
FEATURES = SIM_COLS + list(MISSING_FLAGS)


def features(v: pd.DataFrame) -> pd.DataFrame:
    """Matricea de caracteristici: scorurile (lipsă -> 0) și indicatorii de lipsă."""
    X = v[SIM_COLS].astype(float).fillna(0.0)
    for flag, col in MISSING_FLAGS.items():
        X[flag] = v[col].isna().astype(float)
    return X[FEATURES]


def brand_split(brands: pd.Series, test_size: float = 0.3, seed: int = SEED):
    """Împarte rândurile după marcă: toate rândurile unei mărci ajung fie la antrenare, fie la test.
    Returnează mulțimea mărcilor de test."""
    u = pd.Series(brands.unique())
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(gss.split(u, groups=u))
    return set(u.iloc[test_idx])


def model_grid():
    """Cele trei modele supervizate și valorile hiperparametrilor încercate (alese prin validare încrucișată)."""
    return {
        "regresie logistică": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                               {"logisticregression__C": [0.1, 1.0, 10.0]}),
        "SVM (RBF)": (make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale")),
                      {"svc__C": [0.3, 1.0, 3.0]}),
        "rețea neuronală": (make_pipeline(StandardScaler(),
                                          MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True,
                                                        max_iter=500, random_state=SEED)),
                            {"mlpclassifier__alpha": [1e-2, 1e-3, 1e-4]}),
    }
# Valorile fiecărei grile sunt ordonate de la modelul cel mai simplu (regularizare puternică) la cel mai complex.

TOLERANCE = 1e-3


def _simplest_within_tolerance(cv_results) -> int:
    """Regula de alegere: dintre valorile cu scorul mediu la cel mult TOLERANCE de cel mai bun, prima din grilă,
    adică modelul cel mai simplu. Scorurile aproape egale nu mai depind de zgomotul numeric dintre rulări."""
    scores = np.asarray(cv_results["mean_test_score"])
    return int(np.flatnonzero(scores >= scores.max() - TOLERANCE)[0])


def fit_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_folds: int = 3) -> dict:
    """Alege hiperparametrii prin validare încrucișată pe grupuri (mărci), după aria sub curba precizie–recall
    (cel mai simplu model aflat la cel mult TOLERANCE de cel mai bun scor), apoi reantrenează pe tot setul de antrenare. SVM-ul este calibrat (Platt), ca să dea probabilități."""
    fitted, chosen = {}, {}
    cv = GroupKFold(n_splits=n_folds)
    for name, (est, grid) in model_grid().items():
        gs = GridSearchCV(est, grid, scoring="average_precision", cv=cv, n_jobs=-1,
                          refit=_simplest_within_tolerance)
        gs.fit(X, y, groups=groups)
        chosen[name] = {**gs.best_params_, "scoruri CV": dict(zip([str(list(p_.values())[0]) for p_ in gs.cv_results_["params"]],
                                                              [round(float(x), 4) for x in gs.cv_results_["mean_test_score"]]))}
        best = gs.best_estimator_
        if name.startswith("SVM"):
            # Platt: o regresie logistică pe scorul SVM, potrivită prin validare încrucișată pe mărci
            best = CalibratedClassifierCV(best, method="sigmoid", cv=list(cv.split(X, y, groups)), ensemble=False)
            best.fit(X, y)
        fitted[name] = best
    return fitted, chosen


# ---------------------------------------------------------------------------------------------
# Evaluarea
# ---------------------------------------------------------------------------------------------
def decide(v: pd.DataFrame, prob: np.ndarray, product_class: pd.Series, threshold: float = 0.5,
           exclusive: bool = False) -> pd.DataFrame:
    """Decizia pe rânduri, la nivelul clasei de produs (codurile aceleiași cantități ambalate diferit).
    Pentru fiecare rând se alege clasa cu probabilitatea cea mai mare, iar rândul e legat dacă ea >= prag.
    exclusive=True (Fellegi–Sunter unu-la-unu): candidații unui rând se exclud reciproc, deci probabilitatea
    clasei este suma probabilităților codurilor ei; altfel (modelele supervizate) este maximul lor.
    product_class: clasa fiecărui produs din Nomenclator, indexată după ib."""
    d = v[["ia", "ib"]].assign(prob=prob, clasa=product_class.reindex(v.ib).to_numpy())
    agg = d.groupby(["ia", "clasa"]).agg(prob=("prob", "sum" if exclusive else "max"), ib=("ib", "first"))
    agg = agg.reset_index()
    best = agg.loc[agg.groupby("ia").prob.idxmax()].set_index("ia")
    best["legat"] = best.prob >= threshold
    return best


def row_metrics(best: pd.DataFrame, true: pd.DataFrame, all_rows) -> dict:
    """Metrici pe rânduri. true: perechile corecte (ia, ib); all_rows: toți indecșii rândurilor evaluate
    (inclusiv cele fără niciun candidat, care rămân automat „fără pereche”).
      precizie = legăturile corecte / legăturile făcute
      recall   = legăturile corecte / rândurile care au pereche
      acuratețe = rândurile tratate corect (legate corect sau corect lăsate fără pereche) / toate rândurile"""
    true_set = set(zip(true.ia, true.ib))
    has_pair = set(true.ia)
    b = best.reindex(list(all_rows))
    linked = b.legat.fillna(False).astype(bool)
    correct = pd.Series([(ia, ib) in true_set for ia, ib in zip(b.index, b.ib)], index=b.index)
    tp = (linked & correct).sum()
    prec = tp / linked.sum() if linked.sum() else np.nan
    rec = tp / len(has_pair & set(all_rows))
    no_pair_ok = (~linked & ~b.index.isin(list(has_pair))).sum()
    return {"precizie": prec, "recall": rec, "F1": 2 * prec * rec / (prec + rec),
            "acuratețe pe rânduri": (tp + no_pair_ok) / len(b),
            "rânduri legate greșit": int((linked & ~correct).sum()),
            "rânduri cu pereche nelegate": int(len(has_pair & set(all_rows)) - tp)}


def pair_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    return {"AP (perechi)": average_precision_score(y, prob), "Brier (perechi)": brier_score_loss(y, prob)}


def bootstrap_ci(best: pd.DataFrame, true: pd.DataFrame, all_rows, n_boot: int = 500, seed: int = SEED,
                 metric: str = "F1") -> tuple:
    """Interval de încredere 95% pentru o metrică pe rânduri, prin bootstrap pe rânduri."""
    rng = np.random.default_rng(seed)
    rows = np.array(list(all_rows))
    true_set = set(zip(true.ia, true.ib))
    has_pair = set(true.ia)
    b = best.reindex(rows)
    linked = b.legat.fillna(False).astype(bool).to_numpy()
    correct = np.array([(ia, ib) in true_set for ia, ib in zip(b.index, b.ib)])
    hp = np.isin(rows, list(has_pair))
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(rows), len(rows))
        tp = (linked[i] & correct[i]).sum()
        prec, rec = tp / max(linked[i].sum(), 1), tp / max(hp[i].sum(), 1)
        vals.append({"precizie": prec, "recall": rec, "F1": 2 * prec * rec / (prec + rec)}[metric])
    return tuple(np.percentile(vals, [2.5, 97.5]))
