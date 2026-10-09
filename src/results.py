"""Rezultatele principale ale pipeline-ului, adunate într-un singur fișier: data/processed/rezultate.json.

Fiecare notebook își scrie secțiunea lui (normalizare, potrivire, evaluare, preturi) la final, iar
dashboard-ul (src/dashboard.py) citește cifrele de aici, nu le are scrise de mână. Astfel, pe alte date,
dashboard-ul arată automat rezultatele noi.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _plain(x, digits: int = 4):
    """Transformă valorile numpy/pandas în tipuri JSON simple; numerele reale sunt rotunjite."""
    if isinstance(x, pd.DataFrame):
        return {str(i): _plain(r.to_dict(), digits) for i, r in x.iterrows()}
    if isinstance(x, pd.Series):
        return {str(k): _plain(v, digits) for k, v in x.items()}
    if isinstance(x, dict):
        return {str(k): _plain(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_plain(v, digits) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), digits)
    return x


def save(path: Path, section: str, values: dict) -> dict:
    """Scrie (sau înlocuiește) o secțiune în fișierul de rezultate; celelalte secțiuni rămân neschimbate."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[section] = _plain(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data[section]


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
