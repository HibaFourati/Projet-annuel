from icrawler.builtin import BingImageCrawler
from PIL import Image
import os, time, hashlib, random, shutil
from pathlib import Path


DATA_DIR = Path("dataset")
TARGET_IMAGES = 200
MIN_SIZE = 250

INSTRUMENTS = {
    "piano": [
        "piano keyboard", "grand piano black", "upright piano wood",
        "digital piano modern", "piano keys closeup", "baby grand piano",
        "concert piano stage", "piano interior strings", "piano hammers mechanism",
        "piano player hands", "piano sheet music", "piano bench",
        "piano pedals", "piano soundboard", "piano tuning",
        "vintage piano", "white piano", "electric piano",
        "piano performance", "piano soloist", "piano workshop"
    ],
    "batterie": [
        "drum kit", "snare drum", "bass drum", "rock drum set",
        "jazz drum kit", "electronic drum pad", "drummer playing drums",
        "drum solo", "orchestral percussion", "concert drums",
        "drum sticks closeup", "drum cymbals", "drum studio",
        "drum performance", "drum rehearsal", "drum set closeup", "rock drummer"
    ],
    "harpe": [
        "concert harp", "pedal harp", "harp strings", "harp player",
        "folk harp", "golden harp", "grand harp", "harp soloist",
        "harp interior", "harp tuning", "harp classical", "harp performance",
        "harp closeup", "harp musician", "harp practice"
    ]
}


def get_current_counts():
    counts = {}
    for instrument in INSTRUMENTS:
        folder = DATA_DIR / instrument
        counts[instrument] = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
    return counts

def create_folders():
    DATA_DIR.mkdir(exist_ok=True)
    for instrument in INSTRUMENTS:
        (DATA_DIR / instrument).mkdir(exist_ok=True)


def download_images_for_instrument(instrument, keywords, target_count):
    save_path = DATA_DIR / instrument
    current_count = len(list(save_path.glob("*.jpg")))
    if current_count >= target_count:
        print(f"✓ {instrument}: déjà {current_count} images, rien à télécharger")
        return 0

    images_needed = target_count - current_count
    temp_dir = DATA_DIR / f"temp_{instrument}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    downloaded_total = 0
    random.shuffle(keywords)

    print(f"\nTéléchargement pour {instrument.upper()} ({images_needed} images manquantes)")

    for word in keywords:
        if images_needed <= 0:
            break
        
        for f in temp_dir.glob("*"):
            f.unlink()
        try:
            to_download = min(50, images_needed + 20)
            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_dir)},
                downloader_threads=2,
                parser_threads=1
            )
            crawler.crawl(keyword=word, max_num=to_download, min_size=(MIN_SIZE, MIN_SIZE), filters={'type':'photo'})
            time.sleep(random.uniform(3,6))
            downloaded_files = list(temp_dir.glob("*"))
            valid_added = 0
            for img_file in downloaded_files:
                if images_needed <= 0:
                    break
                try:
                    with Image.open(img_file) as img:
                        if img.width >= MIN_SIZE and img.height >= MIN_SIZE:
                            timestamp = int(time.time() * 1000)
                            random_num = random.randint(1000, 9999)
                            new_name = f"{instrument}_{timestamp}_{random_num}.jpg"
                            img.convert("RGB").save(save_path / new_name, "JPEG", quality=90)
                            images_needed -= 1
                            valid_added += 1
                            downloaded_total += 1
                except:
                    continue
            if valid_added > 0:
                time.sleep(random.uniform(3,6))
        except Exception as e:
            time.sleep(random.uniform(5,10))
            continue

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    new_count = len(list(save_path.glob("*.jpg")))
    print(f"{instrument}: {current_count} → {new_count} images")
    return downloaded_total

def smart_download():
    create_folders()
    current_counts = get_current_counts()
    total_downloaded = 0
    for instrument, count in current_counts.items():
        if count < TARGET_IMAGES:
            downloaded = download_images_for_instrument(instrument, INSTRUMENTS[instrument], TARGET_IMAGES)
            total_downloaded += downloaded
    return total_downloaded


def main():
    print("\nÉtat actuel du dataset :")
    current_counts = get_current_counts()
    for inst, count in current_counts.items():
        print(f"{inst:10} : {count} images")
    input("\nAppuyez sur Entrée pour lancer le téléchargement des images manquantes...")
    downloaded = smart_download()
    print(f"\nTotal téléchargé : {downloaded} nouvelles images")
    print("\nÉtat final du dataset :")
    final_counts = get_current_counts()
    for inst, count in final_counts.items():
        print(f"{inst:10} : {count} images")
    print("\nDataset prêt pour l'entraînement !")

if __name__ == "__main__":
    main()
