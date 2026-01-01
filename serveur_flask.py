from flask import Flask, render_template, request
from pathlib import Path
import numpy as np

# ===============================
# IMPORTS DES MODÈLES
# ===============================
from test_Dataset_SVM import CorrectSVM, extract_robust_features
from test_Dataset_PMC import PMCClassifier, normalize_data, extract_features
from test_Dataset_Modele_Lineaire import (
    Classifier,
    normaliser,
    transform_features,
    extraire_features,
    charger_dataset
)

app = Flask(__name__)

# ===============================
# CONSTANTES
# ===============================
classes = ['guitare', 'piano', 'violon']
names = ['Guitare', 'Piano', 'Violon']

# ===============================
# 1️⃣ ENTRAÎNEMENT DES MODÈLES
# ===============================

# ---------- SVM ----------
X_svm, y_svm = [], []
for i, cls in enumerate(classes):
    for img in Path(f"dataset/{cls}").glob("*.[pj][np]g"):
        f = extract_robust_features(img)
        if f is not None:
            X_svm.append(f)
            y_svm.append(i)

X_svm = np.array(X_svm)
y_svm = np.array(y_svm)

# Normalisation robuste
p1 = np.percentile(X_svm, 1, axis=0)
p99 = np.percentile(X_svm, 99, axis=0)
X_svm_norm = np.clip((X_svm - p1) / (p99 - p1 + 1e-8), 0, 1)

svm_models = []
for c in range(3):
    y_bin = np.where(y_svm == c, 1.0, -1.0)
    svm = CorrectSVM(n_inputs=X_svm.shape[1], learning_rate=0.02, c=1.0)
    svm.fit(X_svm_norm, y_bin)
    svm_models.append(svm)

print("✅ SVM prêt")

# ---------- PMC ----------
X_pmc, y_pmc = [], []
for cls, label in [('guitare', 1), ('piano', -1), ('violon', 0)]:
    for img in Path(f"dataset/{cls}").glob("*.[pj][np]g"):
        f = extract_features(img)
        if f is not None:
            X_pmc.append(f)
            y_pmc.append(label)

X_pmc = np.array(X_pmc)
y_pmc = np.array(y_pmc)

X_pmc_norm, _, pmc_norm = normalize_data(X_pmc, X_pmc)

pmc_model = PMCClassifier(n_inputs=X_pmc_norm.shape[1])
pmc_model.fit(X_pmc_norm, y_pmc)

print("✅ PMC prêt")

# ---------- MODÈLE LINÉAIRE ----------
X_lin, y_lin = charger_dataset()
X_lin_norm, _, lin_norm = normaliser(X_lin, X_lin)

X_lin_t = transform_features(X_lin_norm)
linear_model = Classifier(input_dim=X_lin_t.shape[1])
linear_model.fit(X_lin_norm, y_lin)

print("✅ Modèle linéaire prêt")

# ===============================
# 2️⃣ ROUTES FLASK
# ===============================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    model_type = request.form.get("model_type")
    file = request.files.get("file")

    if not file:
        return render_template("index.html", result="Aucune image envoyée")

    # Sauvegarde temporaire
    temp = Path("temp")
    temp.mkdir(exist_ok=True)
    img_path = temp / file.filename
    file.save(img_path)

    try:
        # -------- SVM --------
        if model_type == "svm":
            f = extract_robust_features(img_path)
            if f is None:
                raise ValueError("Impossible d'extraire les features SVM")
            f_norm = np.clip((f - p1) / (p99 - p1 + 1e-8), 0, 1)
            scores = [m.predict_score(f_norm.reshape(1, -1))[0] for m in svm_models]
            pred = names[int(np.argmax(scores))]

        # -------- PMC --------
        elif model_type == "pmc":
            f = extract_features(img_path)
            if f is None:
                raise ValueError("Impossible d'extraire les features PMC")
            median, iqr = pmc_norm
            f_norm = np.clip((f - median) / (iqr + 1e-8), -5, 5)
            pred_code = pmc_model.predict(f_norm.reshape(1, -1))[0]
            pred = pmc_model.names[pred_code]

        # -------- LINÉAIRE --------
        elif model_type == "linear":
            f = extraire_features(img_path)
            if f is None:
                raise ValueError("Impossible d'extraire les features linéaires")
            median, iqr = lin_norm
            f_norm = (f - median) / (iqr + 1e-8)
            pred_idx = linear_model.predict(f_norm.reshape(1, -1))[0]
            pred = linear_model.names[pred_idx]

        else:
            pred = "Modèle inconnu"

    except Exception as e:
        pred = f"Erreur : {e}"

    return render_template("index.html", result=pred)

# ===============================
# 3️⃣ LANCEMENT DU SERVEUR
# ===============================

if __name__ == "__main__":
    app.run(debug=True)
