# Charger_dataset.py
import os
import cv2
import numpy as np
import zipfile

zip_path = "D:\\projet_etape_3\\instruments_dataset.zip"
extract_path = "D:\\projet_etape_3\\instruments_dataset"

# Décompresser le zip si nécessaire
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("D:\\projet_etape_3")
    print("Dataset décompressé dans :", extract_path)

# Ajuster le chemin si le zip contient déjà un dossier instruments_dataset
dataset_folder = os.path.join(extract_path, "instruments_dataset")
if os.path.exists(dataset_folder):
    extract_path = dataset_folder

# Parcourir les sous-dossiers
X = []
y = []
labels = os.listdir(extract_path)
labels_dict = {}
for idx, label in enumerate(labels):
    folder = os.path.join(extract_path, label)
    if os.path.isdir(folder):
        labels_dict[label] = idx
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                X.append(img.flatten())
                y.append(idx)  # encode directement en entier

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"Dossiers trouvés : {labels}")
print(f"Dataset chargé : {X.shape} images, {len(labels)} classes")
print("Labels :", labels_dict)
