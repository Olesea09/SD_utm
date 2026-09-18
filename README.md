# Potrivirea probabilistă a înregistrărilor între registrele farmaceutice oficiale ale Republicii Moldova

Proiect la disciplina *Știința datelor* (UTM, grupa SD251M) — Olesea Popa.
Constituie Partea I (cercetare) a tezei de master *„Instrument digital bazat pe metode de știința datelor pentru analiza pieței farmaceutice și compararea medicamentelor echivalente”*.

## Problema
Listele CNAM de medicamente compensate nu conțin codul medicamentului, deci nu pot fi legate direct de Nomenclatorul de stat. Proiectul construiește un pipeline reproductibil care leagă sursele prin potrivire probabilistă și analizează dispersia prețurilor între medicamentele echivalente.

## Date (publice, oficiale)
| Fișier `data/raw/` | Sursă | Versiune |
|---|---|---|
| `nomenclator_amdm.xls` | AMDM — https://nomenclator.amdm.gov.md | salvat 02.06.2026 |
| `catalog_preturi_amdm.xls` | AMDM — Catalogul național de prețuri de producător | ord. Rg04-253 / Rg04-254 din 26.08.2026 |
| `cnam_compensate_RO.xlsx`, `cnam_compensate_RU.xlsx` | CNAM — https://cnam.md, lista denumirilor comerciale compensate | 21.08.2026 |

Materialele AMDM sunt publicate sub licența CC BY-SA 4.0. Fișierele din `data/raw/` nu se modifică niciodată.

## Structură
```
data/raw/          date originale, needitate
data/interim/      date curățate (generate)
src/ingest.py      pasul 1 — citirea surselor
src/clean.py       pasul 2 — curățarea de bază, preț pe unitate, grupuri de echivalență
notebooks/01_eda.ipynb   analiza exploratorie (cu constatări)
reports/figures/   figuri generate
run_pipeline.py    rulează totul de la zero
```

## Rulare
```bash
pip install -r requirements.txt
python run_pipeline.py
```
