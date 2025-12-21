# Charger_dataset.py
import os
import cv2
import numpy as np
import zipfile
from sklearn.preprocessing import LabelEncoder

ZIP_PATH = "D:/projet_etape_3/instruments_dataset.zip"
EXTRACT_PATH = "D:/projet_etape_3/instruments_dataset"

# Décompression
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("D:/projet_etape_3")

# Ajustement si double dossier
if os.path.exists(os.path.join(EXTRACT_PATH, "instruments_dataset")):
    EXTRACT_PATH = os.path.join(EXTRACT_PATH, "instruments_dataset")

X, y = [], []

labels = os.listdir(EXTRACT_PATH)
print("Dossiers trouvés :", labels)

for label in labels:
    folder = os.path.join(EXTRACT_PATH, label)
    if os.path.isdir(folder):
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                X.append(img.flatten())
                y.append(label)

X = np.array(X)

# Encodage labels
le = LabelEncoder()
y = le.fit_transform(y)

print(f"Dataset chargé : {X.shape[0]} images, {len(le.classes_)} classes")
print("Labels :", dict(zip(le.classes_, range(len(le.classes_)))))
