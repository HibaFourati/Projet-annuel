from icrawler.builtin import BingImageCrawler
from PIL import Image
import os
import time
import hashlib
import random
from pathlib import Path
import shutil

DATA_DIR = Path("instruments_dataset")
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
        "piano factory", "piano repair", "piano teacher",
        "piano competition", "piano studio", "piano showroom"
    ],
    "guitare": [
        "acoustic guitar wood", "electric guitar rock", "classical guitar nylon",
        "bass guitar metal", "guitar strings closeup", "guitar fretboard",
        "guitar bridge saddle", "guitar headstock tuners", "guitar pickguard",
        "guitar player concert", "guitar amplifier", "guitar effects pedal",
        "guitar case hard", "guitar strap leather", "guitar pick collection",
        "guitar workshop", "guitar luthier", "guitar vintage 1960",
        "guitar custom handmade", "guitar wall hanger", "guitar lesson",
        "guitar solo", "guitar chords", "guitar fingerstyle"
    ],
    "violon": [
        "violin bow hair", "violin strings closeup", "violin f-holes",
        "violin scroll carving", "violin bridge maple", "violin tailpiece",
        "violin chin rest", "violin shoulder rest", "violin case velvet",
        "violin player orchestra", "violin rosin cake", "violin tuning pegs",
        "violin fingerboard ebony", "violin purfling", "violin varnish",
        "violin maker workshop", "violin antique stradivarius", "violin practice",
        "violin concerto", "violin quartet", "violin sheet music",
        "violin teacher", "violin student", "violin masterclass"
    ]
}


def get_current_counts():
    counts = {}
    for instrument in INSTRUMENTS:
        folder = DATA_DIR / instrument
        if folder.exists():
            count = len(list(folder.glob("*.jpg")))
            counts[instrument] = count
        else:
            counts[instrument] = 0
    return counts


def create_folders():
    DATA_DIR.mkdir(exist_ok=True)
    for instrument in INSTRUMENTS:
        (DATA_DIR / instrument).mkdir(exist_ok=True)


def download_images_for_instrument(instrument, keywords, target_count):
    print(f"COMPLÉTION pour: {instrument.upper()}")

    
    save_path = DATA_DIR / instrument
    current_count = len(list(save_path.glob("*.jpg")))
    
    print(f"Images actuelles: {current_count}")
    print(f"Objectif: {target_count}")
    
    if current_count >= target_count:
        print(f"✓ Déjà atteint l'objectif de {target_count} images")
        return 0
    
    images_needed = target_count - current_count
    print(f"Images nécessaires: {images_needed}")
    

    temp_dir = DATA_DIR / f"temp_{instrument}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    

    random.shuffle(keywords)
    
    downloaded_total = 0
    
    for i, word in enumerate(keywords):
        if images_needed <= 0:
            break
        
        print(f"\n  [{i+1}/{len(keywords)}] Mot-clé: '{word}'")

        for f in temp_dir.glob("*"):
            f.unlink()
        
        try:
            to_download = min(50, images_needed + 20)  # Télécharger un peu plus
            
            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_dir)},
                downloader_threads=2,
                parser_threads=1
            )

            crawler.crawl(
                keyword=word,
                max_num=to_download,
                min_size=(MIN_SIZE, MIN_SIZE),
                filters={'type': 'photo'} 
            )
            

            time.sleep(random.uniform(3, 6))
            

            downloaded_files = list(temp_dir.glob("*"))
            print(f"    Téléchargées: {len(downloaded_files)} images")
            

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
                            new_path = save_path / new_name
                    
                            img.convert("RGB").save(new_path, "JPEG", quality=90)
                            
                            images_needed -= 1
                            valid_added += 1
                            downloaded_total += 1
                except Exception as e:
                    continue
            
            print(f"    Images valides ajoutées: {valid_added}")
            print(f"    Restantes à obtenir: {images_needed}")
            
        
            if valid_added > 0:
                time.sleep(random.uniform(5, 8))
            
        except Exception as e:
            print(f" Erreur: {str(e)[:100]}...")
       
            time.sleep(random.uniform(10, 15))
            continue
    

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    new_count = len(list(save_path.glob("*.jpg")))
    print(f"\n   {instrument}: {current_count} → {new_count} images")
    
    return downloaded_total


def smart_download():

    
    create_folders()
    

    current_counts = get_current_counts()
    
    instruments_to_complete = []
    
    for instrument, count in current_counts.items():
        needed = max(0, TARGET_IMAGES - count)
        if needed > 0:
            instruments_to_complete.append((instrument, needed))
            print(f"{instrument:10} : {count:3d} images, besoin de {needed:3d} de plus")
    
    if not instruments_to_complete:
        return
    instruments_to_complete.sort(key=lambda x: x[1], reverse=True)
    
    total_downloaded = 0
    
    for instrument, needed in instruments_to_complete:
        if needed <= 0:
            continue

        keywords = INSTRUMENTS[instrument]
        downloaded = download_images_for_instrument(instrument, keywords, TARGET_IMAGES)
        total_downloaded += downloaded
        

        if instrument != instruments_to_complete[-1][0]:
            time.sleep(random.uniform(10, 20))
    
    return total_downloaded


def advanced_clean_dataset():

    for instrument in INSTRUMENTS:
        folder = DATA_DIR / instrument
        if not folder.exists():
            continue
        
        print(f"\nClasse: {instrument}")
        
        image_files = list(folder.glob("*.jpg"))
        initial_count = len(image_files)
        
        if initial_count == 0:
            continue
        
        print(f"  Avant nettoyage: {initial_count} images")
        

        valid_images = []
        removed_count = 0
        hashes = set()
        
        for i, img_file in enumerate(image_files):
    
            if i % 50 == 0 and i > 0:
                print(f"    Traitées: {i}/{initial_count}")
            
            try:
                
                with Image.open(img_file) as img:
                  
                    if img.width < MIN_SIZE or img.height < MIN_SIZE:
                        raise ValueError("Image trop petite")
                    
                
                    aspect_ratio = img.width / img.height
                    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                        raise ValueError("Ratio d'aspect extrême")
                 
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                
                    img_resized = img.resize((128, 128))
                    img_hash = hashlib.md5(img_resized.tobytes()).hexdigest()
                    
        
                    if img_hash in hashes:
                        raise ValueError("Doublon détecté")
                    
                   
                    hashes.add(img_hash)
                    valid_images.append((img_file, img))
                    
            except Exception as e:
              
                try:
                    os.remove(img_file)
                    removed_count += 1
                except:
                    pass
        
       
        for i, (img_file, img) in enumerate(valid_images):
            new_name = f"{instrument}_{i:04d}.jpg"
            new_path = folder / new_name
            
           
            img.save(new_path, "JPEG", quality=85, optimize=True)

            if img_file.name != new_name:
                try:
                    os.remove(img_file)
                except:
                    pass
        
        final_count = len(list(folder.glob("*.jpg")))
        print(f"  Après nettoyage: {final_count} images")
        print(f"  Images supprimées: {removed_count}")
    
        if final_count > 0:
            print(f"  Diversité (hashs uniques): {len(hashes)}")


def main():

    print("\n ÉTAT ACTUEL DU DATASET:")

    
    current_counts = get_current_counts()
    for instrument, count in current_counts.items():
        status = "✓" if count >= TARGET_IMAGES else ""
        needed = max(0, TARGET_IMAGES - count)
        print(f"{status} {instrument:10} : {count:3d} / {TARGET_IMAGES} ({needed} manquantes)")
    
    total_current = sum(current_counts.values())
    total_target = TARGET_IMAGES * len(INSTRUMENTS)
    print(f"\n Total: {total_current} / {total_target} images")
    

    input("Appuyez sur Entrée pour commencer le téléchargement...")
    
 
    try:
        downloaded = smart_download()
        if downloaded:
            print(f"\n Total téléchargé: {downloaded} nouvelles images")
    except KeyboardInterrupt:
        print("\n\n Téléchargement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n Erreur pendant le téléchargement: {e}")

    advanced_clean_dataset()
    

    
    final_counts = get_current_counts()
    for instrument, count in final_counts.items():
        status = "" if count >= TARGET_IMAGES else " "
        percent = (count / TARGET_IMAGES) * 100
        print(f"{status} {instrument:10} : {count:3d} / {TARGET_IMAGES} ({percent:.1f}%)")
    
    total_final = sum(final_counts.values())
    completion = (total_final / total_target) * 100
    
    print(f"\n OBJECTIF: {TARGET_IMAGES} images par classe")
    print(f" TOTAL: {total_final} / {total_target} images ({completion:.1f}%)")
    

    if total_final < total_target * 0.95: 
        print("\n RECOMMANDATIONS:")
        print("   1. Relancez ce script plusieurs fois")
        print("   2. Ajoutez plus de mots-clés dans les listes")
        print("   3. Essayez à différents moments de la journée")
        print("   4. Vérifiez votre connexion internet")
    
    if total_final >= total_target * 0.9:  # Au moins 90% de l'objectif
        print("\n Dataset prêt pour l'entraînement!")
        print("   Vous pouvez commencer à entraîner votre modèle.")


if __name__ == "__main__":
    main()
