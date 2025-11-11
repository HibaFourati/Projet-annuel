import pandas as pd
import matplotlib.pyplot as plt

# Lire les CSV
xor = pd.read_csv("loss_xor.csv")
linear = pd.read_csv("loss_linear.csv")
dataset = pd.read_csv("loss_dataset.csv")  # ton dataset instruments

# Tracer les courbes
plt.plot(xor["epoch"], xor["loss"], label="XOR")
plt.plot(linear["epoch"], linear["loss"], label="Linéraire")
plt.plot(dataset["epoch"], dataset["loss"], label="Instruments")  # ajout du dataset réel

plt.title("Convergence PMC")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Sauvegarder la figure pour le rapport
plt.savefig("convergence_pmc_tests.png", dpi=300)  # dpi=300 pour une meilleure qualité

# Afficher la figure
plt.show()
