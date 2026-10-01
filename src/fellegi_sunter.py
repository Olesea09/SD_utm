"""Pasul 7a al pipeline-ului: modelul Fellegi–Sunter, fără etichete.

Fiecare pereche candidată este fie corectă (M), fie greșită (U), dar nu știm care. Pentru fiecare câmp k
și fiecare nivel de acord l, modelul are două probabilități:
    m_k(l) = P(nivelul l | pereche corectă),   u_k(l) = P(nivelul l | pereche greșită),
și proporția perechilor corecte, p. Câmpurile sunt presupuse independente în interiorul fiecărei clase,
deci verosimilitatea unei perechi cu nivelurile (l_1, ..., l_K) este
    P(vector | M) = prod_k m_k(l_k),   P(vector | U) = prod_k u_k(l_k).
Un câmp lipsă în una dintre surse nu contribuie (factorul 1). Probabilitatea ca perechea să fie corectă:
    P(M | vector) = p prod m / (p prod m + (1 - p) prod u).

Modelul clasic (em) tratează fiecare pereche separat. Pe datele noastre presupunerea de independență este
încălcată: ambalajele aceleiași mărci au aceeași denumire și firmă, iar EM le pune pe toate în clasa
„corectă”. Varianta cu constrângerea unu-la-unu (em_1to1) cere ca fiecare rând să aibă cel mult o pereche
printre candidații săi, astfel încât candidații concurează între ei (vezi notebook 04).

Două variante de estimare a modelului unu-la-unu:
  - frecventistă: algoritmul EM (Expectation–Maximization) dă o singură valoare pentru rho, m și u;
    incertitudinea lor se obține prin bootstrap (EM reluat pe eșantioane de rânduri cu revenire);
  - bayesiană: probabilitățile au distribuții a priori Beta/Dirichlet, iar eșantionarea Gibbs dă
    distribuția a posteriori a lui rho, m, u și, pentru fiecare pereche, o distribuție a probabilității de potrivire.
"""
import numpy as np
import pandas as pd

ALPHA = 0.5   # pseudo-număr (netezire) la EM: niciun nivel nu primește probabilitatea exactă 0


class Levels:
    """Nivelurile de acord ca matrice one-hot, câte una pentru fiecare câmp (rând de zerouri = lipsă)."""

    def __init__(self, df: pd.DataFrame, cols: list, n_levels: dict = None):
        self.cols = cols
        self.n_levels = n_levels or {c: int(df[c].max()) + 1 for c in cols}
        self.X = {}
        for c in cols:
            v = df[c].to_numpy(dtype="float64", na_value=np.nan)
            X = np.zeros((len(df), self.n_levels[c]))
            ok = ~np.isnan(v)
            X[np.where(ok)[0], v[ok].astype(int)] = 1.0
            self.X[c] = X
        self.n = len(df)

    def logit(self, p, m, u):
        """log [p P(vector|M)] - log [(1-p) P(vector|U)] pentru fiecare pereche."""
        s = np.full(self.n, np.log(p) - np.log1p(-p))
        for c in self.cols:
            s += self.X[c] @ (np.log(m[c]) - np.log(u[c]))
        return s

    def counts(self, w):
        """Pentru fiecare câmp: suma ponderilor w pe fiecare nivel (câmpurile lipsă nu se numără)."""
        return {c: w @ self.X[c] for c in self.cols}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def initial_params(L: Levels, p0: float = 0.05):
    """Punct de plecare pentru EM: m crește cu nivelul (acordul e frecvent la perechile corecte),
    u = frecvențele observate pe toate perechile (aproape toate perechile candidate sunt greșite)."""
    m, u = {}, {}
    for c in L.cols:
        k = L.n_levels[c]
        m[c] = (np.arange(k) + 1.0) ** 3 / ((np.arange(k) + 1.0) ** 3).sum()
        freq = L.X[c].sum(0) + ALPHA
        u[c] = freq / freq.sum()
    return p0, m, u


def em(L: Levels, max_iter: int = 500, tol: float = 1e-8, init=None):
    """Modelul clasic, estimat prin EM: fiecare pereche e tratată separat.
    Returnează (p, m, u, w, istoricul log-verosimilității); w = probabilitatea de potrivire a fiecărei perechi."""
    p, m, u = init or initial_params(L)
    hist = []
    for _ in range(max_iter):
        # pasul E: probabilitatea ca fiecare pereche să fie corectă, cu parametrii actuali
        s = L.logit(p, m, u)
        w = _sigmoid(s)
        # log-verosimilitatea datelor observate (pentru urmărirea convergenței)
        log_u = sum(L.X[c] @ np.log(u[c]) for c in L.cols) + np.log1p(-p)
        hist.append(float(np.sum(log_u + np.logaddexp(0, s))))
        # pasul M: noile probabilități = frecvențele ponderate cu w (corecte) și 1 - w (greșite)
        p = float(np.clip(w.mean(), 1e-6, 1 - 1e-6))
        cm, cu = L.counts(w), L.counts(1 - w)
        m = {c: (cm[c] + ALPHA) / (cm[c] + ALPHA).sum() for c in L.cols}
        u = {c: (cu[c] + ALPHA) / (cu[c] + ALPHA).sum() for c in L.cols}
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2]):
            break
    w = _sigmoid(L.logit(p, m, u))
    return p, m, u, w, hist


def params_table(m: dict, u: dict) -> pd.DataFrame:
    """Tabelul m, u și ponderea log2(m/u) pentru fiecare câmp și nivel."""
    rows = [{"câmp": c.replace("niv_", ""), "nivel": l, "m": m[c][l], "u": u[c][l],
             "pondere log2(m/u)": np.log2(m[c][l] / u[c][l])} for c in m for l in range(len(m[c]))]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Varianta cu constrângerea unu-la-unu (Jaro 1989; Sadinle 2017 pentru varianta bayesiană)
# ---------------------------------------------------------------------------------------------
# Fiecare rând (ex. un rând CNAM) are CEL MULT UN produs corect printre candidații săi. Pentru rândul a,
# cu n_a candidați: cu probabilitatea 1 - rho nu are pereche, iar cu probabilitatea rho perechea este unul
# dintre candidați, fiecare cu probabilitatea a priori 1 / n_a. Candidații aceluiași rând concurează:
#   P(candidatul j este perechea | datele) = (rho / n_a) r_j / [(1 - rho) + (rho / n_a) sum_k r_k],
# unde r_j = P(vector_j | M) / P(vector_j | U) este raportul de verosimilitate al candidatului j.
# Ambalajele aceleiași mărci nu mai pot fi toate „corecte” în același timp.
class Rows:
    """Indexul rândurilor: perechile trebuie să fie ordonate după ia (perechile aceluiași rând consecutive)."""

    def __init__(self, ia):
        ia = np.asarray(ia)
        assert np.all(np.diff(ia) >= 0), "perechile trebuie ordonate după ia"
        self.start = np.r_[0, np.flatnonzero(np.diff(ia)) + 1]
        self.size = np.diff(np.r_[self.start, len(ia)])
        self.row_of = np.repeat(np.arange(len(self.start)), self.size)

    def logsumexp(self, x):
        mx = np.maximum.reduceat(x, self.start)
        return mx + np.log(np.add.reduceat(np.exp(x - mx[self.row_of]), self.start))


def _llr(L: Levels, m, u):
    """log r_j = log P(vector | M) - log P(vector | U) pentru fiecare pereche."""
    return sum(L.X[c] @ (np.log(m[c]) - np.log(u[c])) for c in L.cols)


def _posterior_1to1(llr, rows: Rows, rho):
    """Probabilitatea ca fiecare candidat să fie perechea rândului său și log-verosimilitatea pe rânduri."""
    a = np.log(rho) - np.log(rows.size)[rows.row_of] + llr          # log [(rho / n_a) r_j]
    none = np.full(len(rows.start), np.log1p(-rho))                  # log (1 - rho): rândul nu are pereche
    lse = np.logaddexp(none, rows.logsumexp(a))
    return np.exp(a - lse[rows.row_of]), lse


def em_1to1(L: Levels, rows: Rows, max_iter: int = 1000, tol: float = 1e-9, init=None, rho0: float = 0.9):
    """EM cu constrângerea unu-la-unu. Returnează (rho, m, u, w, istoric); w = P(candidatul e perechea)."""
    _, m, u = init or initial_params(L)
    rho, hist = rho0, []
    for _ in range(max_iter):
        w, lse = _posterior_1to1(_llr(L, m, u), rows, rho)          # pasul E
        hist.append(float(lse.sum()))
        rho = float(np.clip(np.add.reduceat(w, rows.start).mean(), 1e-6, 1 - 1e-6))   # pasul M
        cm, cu = L.counts(w), L.counts(1 - w)
        m = {c: (cm[c] + ALPHA) / (cm[c] + ALPHA).sum() for c in L.cols}
        u = {c: (cu[c] + ALPHA) / (cu[c] + ALPHA).sum() for c in L.cols}
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2]):
            break
    w, _ = _posterior_1to1(_llr(L, m, u), rows, rho)
    return rho, m, u, w, hist


def gibbs_1to1(L: Levels, rows: Rows, n_iter: int = 1500, burn: int = 500, thin: int = 5, seed: int = 42,
               prior_m: float = 1.0, prior_u: float = 1.0, prior_rho=(1.0, 1.0), init=None):
    """Varianta bayesiană cu constrângerea unu-la-unu (în spiritul lui Sadinle, 2017).
    A priori: rho ~ Beta(prior_rho), m_k ~ Dirichlet(prior_m), u_k ~ Dirichlet(prior_u).
    La fiecare iterație: (1) pentru fiecare rând se eșantionează care candidat este perechea, sau niciunul;
    (2) rho ~ Beta(a + rânduri cu pereche, b + rânduri fără); (3) m_k, u_k ~ Dirichlet(prior + numărările).
    Pornește de la soluția EM. Returnează eșantioanele lui rho, m, u și P(candidatul e perechea) pe eșantioane."""
    rng = np.random.default_rng(seed)
    rho, m, u = init if init is not None else em_1to1(L, rows)[:3]
    n_rows = len(rows.start)
    keep = {"rho": [], "m": {c: [] for c in L.cols}, "u": {c: [] for c in L.cols}, "w": []}
    for it in range(n_iter):
        w, _ = _posterior_1to1(_llr(L, m, u), rows, rho)
        # (1) alegerea perechii fiecărui rând: P(niciunul) = 1 - suma w pe rând
        cum = np.cumsum(w)
        before = np.r_[0.0, cum][rows.start]                          # suma cumulată înaintea fiecărui rând
        draw = rng.random(n_rows)
        target = before + draw                                        # dacă draw < suma w a rândului: alegem un candidat
        has = draw < np.add.reduceat(w, rows.start)
        pick = np.searchsorted(cum, target[has], side="right")
        z = np.zeros(L.n)
        z[pick] = 1.0
        # (2) și (3)
        rho = rng.beta(prior_rho[0] + has.sum(), prior_rho[1] + n_rows - has.sum())
        cm, cu = L.counts(z), L.counts(1 - z)
        m = {c: rng.dirichlet(prior_m + cm[c]) for c in L.cols}
        u = {c: rng.dirichlet(prior_u + cu[c]) for c in L.cols}
        m = {c: np.clip(v, 1e-12, None) / np.clip(v, 1e-12, None).sum() for c, v in m.items()}
        u = {c: np.clip(v, 1e-12, None) / np.clip(v, 1e-12, None).sum() for c, v in u.items()}
        if it >= burn and (it - burn) % thin == 0:
            keep["rho"].append(rho)
            for c in L.cols:
                keep["m"][c].append(m[c]); keep["u"][c].append(u[c])
            keep["w"].append(_posterior_1to1(_llr(L, m, u), rows, rho)[0].astype("float32"))
    return {"rho": np.array(keep["rho"]), "m": {c: np.array(v) for c, v in keep["m"].items()},
            "u": {c: np.array(v) for c, v in keep["u"].items()}, "w": np.column_stack(keep["w"])}


def bootstrap_1to1(df: pd.DataFrame, cols: list, n_boot: int = 50, seed: int = 42):
    """Incertitudinea frecventistă: EM unu-la-unu reluat pe n_boot eșantioane de rânduri, cu revenire.
    Se reeșantionează rândurile (ia), nu perechile, pentru că perechile aceluiași rând nu sunt independente.
    Returnează un tabel lung (eșantion, câmp, nivel, m, u) și valorile lui rho."""
    rng = np.random.default_rng(seed)
    n_levels = {c: int(df[c].max()) + 1 for c in cols}
    groups = df.groupby("ia").indices
    keys = np.array(list(groups))
    out, rhos = [], []
    for b in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        new_ia = np.repeat(np.arange(len(pick)), [len(groups[keys[i]]) for i in pick])   # rândurile repetate devin distincte
        Lb = Levels(df.iloc[idx], cols, n_levels)
        rho, m, u, _, _ = em_1to1(Lb, Rows(new_ia))
        rhos.append(rho)
        out += [(b, c, l, m[c][l], u[c][l]) for c in cols for l in range(n_levels[c])]
    return pd.DataFrame(out, columns=["esantion", "camp", "nivel", "m", "u"]), np.array(rhos)
