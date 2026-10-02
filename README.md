# Zweistufiges Experiment zur Schätzunsicherheit

Diese grobe Python-Implementierung übersetzt den bisherigen Versuchsplan in zwei getrennte, aber aufeinander abgestimmte Stufen:

1. **Kontrollierte Monte-Carlo-Analyse:** Die wahre Kovarianzmatrix ist bekannt. Sample Covariance und Ledoit-Wolf werden auf exakt denselben Zufallsziehungen verglichen. Gemessen werden Kovarianzschätzfehler, Gewichtsfehler und der Anstieg des wahren Portfoliorisikos.
2. **Realistische Rolling-Window-Analyse:** Auf einem Aktienpanel werden monatlich Sample-GMV, Ledoit-Wolf-GMV und 1/N neu gebildet. Gemessen werden Out-of-Sample-Volatilität, Turnover, Sharpe Ratio und kumuliertes Vermögen.

Alle vorläufig festgelegten Entscheidungen sind in `portfolio_experiment/config.py` mit `TODO(TBD)` markiert. Die Implementierung soll Orientierung geben und ist noch kein final eingefrorenes Forschungsdesign.

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Schnellstart: kontrollierte Analyse

Ein kleiner Smoke Run (wenige Sekunden):

```powershell
python run_experiment.py controlled --quick
```

Der größere, in der Konfiguration hinterlegte Lauf:

```powershell
python run_experiment.py controlled
```

Ergebnisse werden standardmäßig unter `outputs/controlled/` abgelegt:

- `replications.csv`: eine Zeile je Konfiguration, Replikation und Schätzer
- `summary.csv`: Mittelwerte und Streuungen nach Konfiguration
- `prial.csv`: relative Verbesserung des Kovarianzverlusts durch Shrinkage
- mehrere PNG-Abbildungen für die zentralen Zusammenhänge

## Realistische Analyse

Die vorhandene Feather-Datei kann so verwendet werden:

```powershell
python run_experiment.py realistic --data "..\PortOpt\Data\US_datastream\US_data_panel_filtered_0.15.feather"
```

Wichtig: Die Datei ist sehr groß. Der realistische Lauf wurde deshalb nicht vollständig auf dem gesamten Datensatz ausgeführt. Für einen ersten Test empfiehlt sich eine kleinere, repräsentative Feather- oder CSV-Datei mit den Spalten `Date`, `DSCD`, `Return` und optional `MarketCAP`.

## Zentrale Annahmen (vorläufig)

- Kontrollierte Analyse: unbeschränktes GMV mit Vollinvestition; bei singulärer Sample-Kovarianz wird die Moore-Penrose-Pseudoinverse verwendet.
- DGPs: Identitätsmatrix als Sanity Check und eine einfache Ein-Faktor-Korrelationsstruktur als nicht-triviale Alternative.
- Informationsstatus: `c = N/T`; die Standardwerte liegen beidseits des kritischen Bereichs `c = 1`.
- Realistische Analyse: long-only, voll investiert, 504 Handelstage Lookback, 21 Tage Halteperiode und Auswahl nach Marktkapitalisierung.
- Transaktionskosten werden noch nicht vom Return abgezogen; Turnover wird bereits gemessen. Das ist im Code als offene Erweiterung markiert.

## Struktur

```text
portfolio_experiment/
  config.py       # alle Designparameter und TBDs
  dgp.py          # bekannte Kovarianzstrukturen und Simulation
  estimators.py   # Sample und Ledoit-Wolf
  optimization.py # GMV-Löser
  metrics.py      # Fehler- und Performance-Maße
  controlled.py   # Stufe 1
  realistic.py    # Stufe 2
run_experiment.py # Kommandozeilen-Einstieg
tests/            # kleine Plausibilitätstests
```

