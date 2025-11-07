use image::io::Reader as ImageReader;
use std::fs;

fn main() {
    let classes = vec!["guitare", "piano", "violon"];

    for class_name in classes {
        let path = format!("dataset/{}", class_name);

        let entries = match fs::read_dir(&path) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Erreur en lisant le dossier {}: {}", path, e);
                continue;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Erreur sur un fichier: {}", e);
                    continue;
                }
            };

            let img_path = entry.path();

            if let Some(ext) = img_path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "jpg" || ext_lower == "jpeg" {
                    // Ouvrir et décoder séparément
                    let img = match ImageReader::open(&img_path) {
                        Ok(reader) => match reader.decode() {
                            Ok(_) => continue, // image OK, on ne fait rien
                            Err(e) => {
                                println!("Fichier corrompu : {} ({})", img_path.display(), e);
                                continue;
                            }
                        },
                        Err(e) => {
                            println!("Impossible d'ouvrir : {} ({})", img_path.display(), e);
                            continue;
                        }
                    };
                }
            }
        }
    }

    println!("Scan terminé !");
}

