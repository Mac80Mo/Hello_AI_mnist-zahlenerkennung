import tensorflow as tf
import cv2 as cv
import os

# Modell laden
modell_path = "C:/Marcus/GitHubProjekte/Python/Hello_AI_mnist-zahlenerkennung/model/mnist-modell.keras"
modell = tf.keras.models.load_model(modell_path)

# Bildpfad
file_name = "C:/Marcus/GitHubProjekte/Python/Hello_AI_mnist-zahlenerkennung/data/2.png"
file_path = os.path.abspath(file_name)

# Debugging: Pfad prüfen
print(f"Versuchter Pfad zum Bild: {file_path}")

# Bild laden
image = cv.imread(file_name, cv.IMREAD_GRAYSCALE)

# Prüfen, ob das Bild korrekt geladen wurde
if image is None:
    print(f"Fehler: Das Bild {file_name} konnte nicht geladen werden. Überprüfe den Pfad oder die Datei.")
else:
    try:
        # Bild umformen
        image = image.reshape(1, 28 * 28, 1)
        print("Bild erfolgreich geladen und umgeformt.")
        
        # Vorhersage
        erkennung = modell.predict(image, batch_size=1)
        print(f"Erkannte Zahl: {erkennung.argmax()}")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
