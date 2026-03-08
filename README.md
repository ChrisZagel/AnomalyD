# Colab + GitHub Proof of Concept

Dieses Repository zeigt einen **etwas realistischeren Proof of Concept** für Google Colab:
- stabiler Einstiegspunkt über `main.py`
- modulare Struktur mit `app/` (vorbereitet für Wachstum)
- echte Abhängigkeit über `requirements.txt` (`rich`)
- optionale Ausgabe von Laufzeitdaten als JSON-Datei

## Projektstruktur

```text
.
├── app/
│   ├── __init__.py
│   ├── cli.py
│   ├── models.py
│   └── services.py
├── tests/
│   └── test_services.py
├── main.py
└── requirements.txt
```

## Was das Programm macht

Beim Start wird eine formatierte Tabelle ausgegeben mit:
- Begrüßung (`Hello <name>!`)
- aktuellem UTC-Zeitstempel
- Python-Version
- Plattform-Information

Zusätzlich kann das Programm dieselben Daten als JSON exportieren.

## In Google Colab ausführen

```python
!git clone https://github.com/user/repo.git
%cd repo
!pip install -r requirements.txt
!python main.py
```

Optional mit Parametern:

```python
!python main.py --name "Max"
!python main.py --name "Max" --json-out runtime.json
!cat runtime.json
```

## Optionaler Testlauf in Colab

```python
!python -m unittest discover -s tests -p "test_*.py"
```
