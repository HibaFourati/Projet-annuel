from PIL import Image
import os
import numpy as np
import csv

# Chemins complets vers mes dossiers
folders = [
    r"C:\Users\foura\Downloads\ESGI_2025-ProjetAnnuel3BigDataJanvier_E1_10-15-17.12.03_Problématique (2)\instruments_dataset\piano",
    r"C:\Users\foura\Downloads\ESGI_2025-ProjetAnnuel3BigDataJanvier_E1_10-15-17.12.03_Problématique (2)\instruments_dataset\guitare",
    r"C:\Users\foura\Downloads\ESGI_2025-ProjetAnnuel3BigDataJanvier_E1_10-15-17.12.03_Problématique (2)\instruments_dataset\violon"
]

# Classe correspondante pour chaque dossier (one-hot)
classes = {
    "piano": [1.0, 0.0, 0.0],
    "guitare": [0.0, 1.0, 0.0],
    "violon": [0.0, 0.0, 1.0]
}

inputs = []
targets = []

# Taille à redimensionner
output_size = (28, 28)

for folder in folders:
    label = os.path.basename(folder)  # 'piano', 'guitare', 'violon'
    for file in os.listdir(folder):
        if file.lower().endswith(".jpg") or file.lower().endswith(".png"):
            path = os.path.join(folder, file)
            img = Image.open(path).convert("L")          # niveaux de gris
            img = img.resize(output_size)                # redimensionne
            vec = np.array(img).flatten() / 255.0       # transforme en vecteur [0,1]
            inputs.append(vec.tolist())
            targets.append(classes[label])

# Sauvegarder dans un fichier CSV
with open("dataset_instruments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for i in range(len(inputs)):
        writer.writerow(inputs[i] + targets[i])

print("CSV généré : dataset_instruments.csv")
