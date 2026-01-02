import socket
import pickle
import numpy as np
from test_dataset_svm import extract_robust_features
from pathlib import Path

# === Choisir une image à tester ===
img_path = Path("dataset/guitare/guitare_0001.jpg")  # CHANGE selon ton dataset
features = extract_robust_features(img_path)

if features is None:
    raise ValueError(f"Impossible d'extraire les features depuis {img_path}")

request = {
    "model_type": "svm",  # "svm", "pmc", "linear"
    "features": features.tolist()
}

HOST = '127.0.0.1'
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
client.sendall(pickle.dumps(request))

response = client.recv(8192)
result = pickle.loads(response)
print("Résultat du serveur :", result)

client.close()
