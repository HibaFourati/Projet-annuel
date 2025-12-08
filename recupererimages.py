from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os
import time
import hashlib
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_DIR = Path("instruments_dataset")
NB_IMAGES = 80
MIN_SIZE = 250

INSTRUMENTS = {
    "piano": [
        "piano à queue",
        "piano droit",
        "pianiste en concert",
        "piano noir",
        "clavier de piano"
    ],
    "guitare": [
        "guitare acoustique",
        "guitare électrique",
        "guitariste sur scène",
        "cordes de guitare",
        "basse électrique"
    ],
    "violon": [
        "violon classique",
        "violon d’orchestre",
        "violon ancien",
        "violoniste",
        "violon et archet"
    ]
}
# -------------------------------


def create_folders():
    """Crée les dossiers du dataset."""
    DATA_DIR.mkdir(exist_ok=True)
    for instrument in INSTRUMENTS:
        (DATA_DIR / instrument).mkdir(exist_ok=True)


def download_images():
    """Télécharge les images Google pour chaque instrument."""
    print(" Démarrage du téléchargement...\n")
    for instrument, keywords in INSTRUMENTS.items():
        print(f" Instrument : {instrument}")
        save_path = DATA_DIR / instrument

        for word in keywords:
            print(f"  🔍 Mot-clé : {word}")
            crawler = GoogleImageCrawler(storage={"root_dir": str(save_path)})
            crawler.crawl(keyword=f"{word} photo", max_num=NB_IMAGES)
    print("\n Téléchargement terminé !")


def image_hash(file_path):
    """Retourne le hash MD5 d’une image (permet de détecter les doublons)."""
    try:
        with Image.open(file_path) as img:
            img = img.convert("RGB").resize((128, 128))  # normalisation
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None


def clean_image(file_path):
    """Vérifie si l'image est valide, trop petite ou transparente."""
    try:
        with Image.open(file_path) as img:
            if img.width < MIN_SIZE or img.height < MIN_SIZE:
                raise ValueError("Image trop petite")
            if img.mode in ("RGBA", "LA") or (
                file_path.suffix.lower() == ".png" and "A" in img.getbands()
            ):
                raise ValueError("Image avec transparence")
    except Exception:
        try:
            os.remove(file_path)
        except Exception:
            pass


def clean_dataset():
    """Nettoie les images invalides et supprime les doublons."""
    print("\n Nettoyage du dataset...\n")

    all_hashes = set()

    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            print(f"   Classe : {folder.name}")
            for img_file in folder.iterdir():
                if not img_file.is_file():
                    continue

                # Nettoyage de base
                clean_image(img_file)
                time.sleep(0.005)

                # Vérification des doublons
                img_hash = image_hash(img_file)
                if not img_hash:
                    continue
                if img_hash in all_hashes:
                    try:
                        os.remove(img_file)
                        print(f"    🗑️ Doublon supprimé : {img_file.name}")
                    except Exception:
                        pass
                else:
                    all_hashes.add(img_hash)

    print("\n Nettoyage terminé ! Aucune image en double.")


def main():
    create_folders()
    download_images()
    clean_dataset()
    print("\n Dataset prêt à être utilisé !")


if __name__ == "__main__":
    main()
