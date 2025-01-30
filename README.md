# MNIST-Zahlenkennung mit TensorFlow und OpenCV

Dieses Projekt implementiert ein KI-Modell zur Erkennung handgeschriebener Ziffern (MNIST-Datensatz). Es beinhaltet sowohl das Training des Modells als auch die Anwendung zur Vorhersage von Ziffern in Bilddateien.

## Projektübersicht

### Ziel des Projekts
Das Ziel dieses Projekts ist es, ein neuronales Netzwerk zu trainieren, das handgeschriebene Ziffern erkennt, und dieses Modell anschließend zu nutzen, um Ziffern aus Bilddateien vorherzusagen.

### Hauptkomponenten
1. **Training des Modells**:
   - Ein Jupyter-Notebook (`mnist_ki_training.ipynb`) trainiert ein neuronales Netzwerk mit dem MNIST-Datensatz.
   - Das trainierte Modell wird im Ordner `model/` als Datei `mnist-modell.keras` gespeichert.

2. **Vorhersage mit dem Modell**:
   - Ein Python-Skript (`zahlenkennung.py`) liest ein Bild (Graustufen, 28x28 Pixel) ein, verarbeitet es und nutzt das trainierte Modell zur Vorhersage der dargestellten Ziffer.

---

## Verzeichnisstruktur

```plaintext
Hello_AI_mnist-zahlenerkennung/
├── data/
│   ├── 1.png          # Beispiel-Bilddateien zur Vorhersage
│   ├── 2.png
│   ├── ...
├── model/
│   ├── mnist-modell.keras  # Trainiertes KI-Modell
├── src/
│   ├── mnist_ki_training.ipynb  # Notebook für das Training
│   ├── zahlenkennung.py         # Skript zur Vorhersage
│   ├── tensoren.py              # Hilfsfunktionen (optional)
├── README.md        # Dokumentation
├── requirements.txt # Abhängigkeiten
└── .gitignore       # Dateien und Ordner, die nicht ins Repository gehören
