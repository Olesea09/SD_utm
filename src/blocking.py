"""Pasul 5 al pipeline-ului: blocarea, adică alegerea perechilor candidate.

Compararea fiecărui rând dintr-o listă cu toate cele 6.242 de produse din Nomenclator ar însemna
milioane de perechi, aproape toate evident diferite. Blocarea păstrează doar perechile care au
o cheie comună, iar modelul de potrivire le compară apoi în detaliu doar pe acestea.

Două chei, unite (o pereche e candidată dacă are cel puțin una dintre ele în comun):
  - dci_stem: rădăcina substanței active („metfor” pentru „Metformini hydrochloridum” și „METFORMINUM”);
  - prefixul denumirii: primele 4 litere ale denumirii normalizate. Este rezerva pentru produsele
    fără DCI utilizabilă („Combinaţie” în Nomenclator) și pentru DCI scrise foarte diferit.
"""
import pandas as pd

PREFIX_LEN = 4


def name_prefix(s: pd.Series, k: int = PREFIX_LEN) -> pd.Series:
    """Primele k litere ale denumirii normalizate („cardiosalin protect” -> „card”); NaN dacă sunt prea puține."""
    p = s.fillna("").str.replace(r"[^a-z]", "", regex=True).str[:k]
    return p.where(p.str.len() >= 3)


def candidate_pairs(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Perechile candidate (ia, ib) dintre rândurile lui a și b, cu cheia care le-a adus („dci”, „nume” sau ambele).
    a și b trebuie să aibă coloanele dci_stem și denumire_norm; ia și ib sunt indecșii lor."""
    parts = []
    for key, col_a, col_b in [("dci", a["dci_stem"], b["dci_stem"]),
                              ("nume", name_prefix(a["denumire_norm"]), name_prefix(b["denumire_norm"]))]:
        ka = pd.DataFrame({"ia": a.index, "k": col_a.values}).dropna()
        kb = pd.DataFrame({"ib": b.index, "k": col_b.values}).dropna()
        parts.append(ka.merge(kb, on="k")[["ia", "ib"]].assign(bloc=key))
    pairs = pd.concat(parts)
    # o pereche adusă de ambele chei apare o singură dată, cu „dci+nume”
    return (pairs.groupby(["ia", "ib"]).bloc.agg(lambda x: "+".join(sorted(set(x))))
                 .reset_index())


def blocking_metrics(pairs: pd.DataFrame, n_a: int, n_b: int, true_pairs: pd.DataFrame = None) -> dict:
    """Rata de reducere = cât din toate perechile posibile (n_a x n_b) elimină blocarea.
    Completitudinea perechilor = ce proporție din perechile corecte rămân după blocare
    (doar dacă adevărul e cunoscut: true_pairs cu coloanele ia, ib)."""
    out = {"perechi candidate": len(pairs),
           "perechi posibile": n_a * n_b,
           "rata de reducere": 1 - len(pairs) / (n_a * n_b),
           "candidați pe rând (median)": pairs.groupby("ia").size().median(),
           "rânduri fără niciun candidat": n_a - pairs.ia.nunique()}
    if true_pairs is not None:
        found = true_pairs.merge(pairs[["ia", "ib"]], on=["ia", "ib"], how="left", indicator=True)
        out["completitudinea perechilor"] = (found["_merge"] == "both").mean()
        # la nivel de rând: rândul are cel puțin o pereche corectă printre candidați
        out["rânduri cu perechea printre candidați"] = (found.assign(ok=found["_merge"] == "both")
                                                        .groupby("ia").ok.any().mean())
    return out
