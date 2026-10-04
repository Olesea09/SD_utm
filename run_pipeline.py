"""Rulează întregul pipeline de la zero: date brute -> date curățate -> notebook-uri executate.

Utilizare:  python run_pipeline.py
Pentru alte date: înlocuiți fișierele din data/raw/ (aceleași nume) și rulați din nou.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ["notebooks/01_eda.ipynb", "notebooks/02_normalizare.ipynb",
             "notebooks/03_semisintetic_blocare_comparare.ipynb", "notebooks/04_potrivire_modele.ipynb",
             "notebooks/05_evaluare_explicabilitate.ipynb",
             "notebooks/06_analiza_preturilor.ipynb"]  # următoarele etape se adaugă aici

for nb in NOTEBOOKS:
    print(f"==> {nb}")
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--ExecutePreprocessor.timeout=1800", "--inplace", str(ROOT / nb)], check=True)
print("Gata. Figurile sunt în reports/figures/, datele curățate în data/interim/.")
