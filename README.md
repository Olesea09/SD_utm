# Integrarea probabilistă a registrelor farmaceutice oficiale din Republica Moldova și analiza prețurilor medicamentelor echivalente

Proiect la disciplina *Știința datelor* (UTM, grupa SD251M) — Olesea Popa.
Constituie Partea I (cercetare) a tezei de master *„Instrument digital bazat pe metode de știința datelor pentru analiza pieței farmaceutice și compararea medicamentelor echivalente”*.

## Problema
Lista CNAM a medicamentelor compensate nu conține codul medicamentului, deci nu poate fi legată direct de Nomenclatorul de stat al medicamentelor (AMDM) și, prin el, de Catalogul național de prețuri. Proiectul construiește un pipeline reproductibil care leagă sursele prin potrivire probabilistă (record linkage), estimează incertitudinea fiecărei legături și analizează pe datele integrate dispersia prețurilor de producător între medicamentele echivalente.

## Date (publice, oficiale)
| Fișier `data/raw/` | Sursă | Versiune |
|---|---|---|
| `nomenclator_amdm.xls` | AMDM — https://nomenclator.amdm.gov.md | salvat 02.06.2026 |
| `catalog_preturi_amdm.xls` | AMDM — Catalogul național de prețuri de producător | ord. Rg04-253 / Rg04-254 din 26.08.2026 |
| `cnam_compensate_RO.xlsx`, `cnam_compensate_RU.xlsx` | CNAM — https://cnam.md, lista denumirilor comerciale compensate | 21.08.2026 |

Materialele AMDM sunt publicate sub licența CC BY-SA 4.0. Fișierele din `data/raw/` nu se modifică niciodată.
Prețurile din Catalog sunt **prețuri de producător** (ex works), nu prețuri de farmacie.

## Pipeline
| Pas | Cod | Notebook | Ce face |
|---|---|---|---|
| 1. Citire | `src/ingest.py` | `01_eda` | citește sursele, numește coloanele uniform |
| 2. Curățare | `src/clean.py` | `01_eda` | curățare de bază, preț pe unitate, grupuri de echivalență |
| 3. Normalizare | `src/normalize.py` | `02_normalizare` | formă, doză, denumire, substanță activă, firmă, divizare — identic pe toate sursele |
| 4. Set semisintetic | `src/semisynthetic.py` | `03_semisintetic_blocare_comparare` | perechi Catalog–Nomenclator cu variațiile de scriere măsurate în CNAM, răspuns cunoscut prin cod |
| 5. Blocare | `src/blocking.py` | `03_…` | perechi candidate (rădăcina DCI sau prefixul denumirii) |
| 6. Comparare | `src/compare.py` | `03_…` | vectorul de comparare: scoruri continue și niveluri de acord |
| 7. Potrivire | `src/fellegi_sunter.py`, `src/models.py` | `04_potrivire_modele` | Fellegi–Sunter (EM clasic, unu-la-unu, bootstrap, Gibbs bayesian); regresie logistică, SVM, rețea neuronală |
| 8. Evaluare | `src/evaluation.py` | `05_evaluare_explicabilitate` | evaluare pe 200 de rânduri CNAM etichetate manual, reponderată pe straturi; SHAP, LIME; alegerea modelului |
| 9. Prețuri | `src/prices.py` | `06_analiza_preturilor` | dispersia prețurilor, compensat vs. necompensat, concurență (efecte fixe + bootstrap; modele bayesiene ierarhice în PyMC) |
| 10. Dashboard | `src/dashboard.py`, `src/results.py` | — | `reports/dashboard.html`: prezentarea rezultatelor (ecrane parcurse cu săgețile) și căutarea medicamentelor echivalente, generată din rezultatele pipeline-ului |

`src/labeling.py` a generat o singură dată eșantionul pentru etichetarea manuală și nu face parte din pipeline. Etichetele se află în `data/labels/` și sunt folosite **doar** la evaluarea finală.

## Rezultate principale
- Normalizarea ridică potrivirea exactă CNAM → Nomenclator de la 69,4% la 91,0%.
- Modelul Fellegi–Sunter clasic confundă produsul cu marca (ambalajele aceleiași mărci încalcă independența condiționată). Cu **constrângerea unu-la-unu**, modelul regăsește fără etichete probabilitățile adevărate ale setului semisintetic (abatere medie 0,001).
- Pe cele 200 de rânduri CNAM etichetate manual, **Fellegi–Sunter unu-la-unu (varianta bayesiană)** are F1 reponderat 0,993 [0,982; 1,000], față de 0,983–0,984 la modelele supervizate, care ratează perechile cu doza scrisă diferit (diferență între datele de antrenare și cele reale).
- Lista integrată: 95,7% dintre rândurile CNAM sunt legate de Nomenclator; 99 sunt marcate pentru verificare manuală; 92,1% dintre rândurile legate au preț în Catalog.
- Între medicamentele echivalente în formă solidă, prețul de producător pe unitate diferă în mediană de 1,3 ori și de cel puțin 2 ori în unul din cinci grupuri. Compensatele sunt în medie cu ~7% mai ieftine decât echivalentele necompensate, fără semnificație la 95%. Dispersia prețurilor crește real cu numărul de firme: dublarea lor o înmulțește cu 1,54 [1,34; 1,78].

Figurile sunt în `reports/figures/` (fig01–fig18). Detaliile, cu cifrele exacte și constatările, se află în notebook-uri, salvate cu rezultatele afișate.

## Rulare
Mediu: Python 3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` rulează notebook-urile 01–06, în ordine, din datele brute, apoi generează dashboard-ul (aproximativ 15 minute). Rezultatele intermediare se scriu în `data/interim/`, iar cele finale în `data/processed/`:
- `cnam_legat_nomenclator.csv`: fiecare rând CNAM cu codul din Nomenclator, probabilitatea legăturii, intervalul credibil și indicatorul de verificare manuală;
- `grupuri_echivalenta_preturi.csv`: statisticile de preț pe grupuri de echivalență;
- `rezultate.json`: cifrele principale ale fiecărui pas, scrise de notebook-uri și citite de dashboard.

Dashboard-ul `reports/dashboard.html` se deschide direct în browser, fără internet (F: ecran complet, N: notițe).

Ambele foldere sunt generate și nu sunt păstrate în git.

**Reproductibilitate.** Toate procesele aleatoare folosesc un seed fix (42), iar hiperparametrii sunt aleși printr-o regulă stabilă. O nouă rulare reproduce aceleași rezultate. Excepția este modelul bayesian „compensat vs. necompensat” din notebook-ul 06, care poate varia între rulări în limita erorii Monte Carlo (câteva zecimi de punct procentual). Pentru alte versiuni ale surselor se înlocuiesc fișierele din `data/raw/` (aceleași nume) și se rulează din nou pipeline-ul. Modelul ales nu are nevoie de etichete noi.

## Limitări
Prețurile sunt de producător, deci nu descriu ce plătește pacientul în farmacie. Analiza prețurilor cuprinde doar formele solide (la lichide prețul ar trebui normalizat pe cantitatea de substanță). Datele provin dintr-un singur moment, deci rezultatele sunt asocieri, nu efecte cauzale.
