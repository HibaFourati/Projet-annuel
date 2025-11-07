use image::{io::Reader as ImageReader};
use std::fs;

pub fn load_dataset(base_path: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let classes = vec!["guitare", "piano", "violon"];
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for (label, class_name) in classes.iter().enumerate() {
        let path = format!("{}/{}", base_path, class_name);

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

            // Vérifier l'extension jpg/jpeg
            if let Some(ext) = img_path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "jpg" || ext_lower == "jpeg" {
                    // Ouvrir et décoder l'image
                    let img = match ImageReader::open(&img_path) {
                        Ok(reader) => match reader.decode() {
                            Ok(i) => i,
                            Err(e) => {
                                eprintln!("Erreur en décodant {}: {}", img_path.display(), e);
                                continue;
                            }
                        },
                        Err(e) => {
                            eprintln!("Erreur en ouvrant {}: {}", img_path.display(), e);
                            continue;
                        }
                    };

                    // Redimensionner à 32x32 et convertir en RGB
                    let resized = img.resize_exact(32, 32, image::imageops::FilterType::Nearest).to_rgb8();

                    // Normaliser les pixels
                    let pixels: Vec<f64> = resized
                        .pixels()
                        .flat_map(|p| vec![p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
                        .collect();

                    x_data.push(pixels);
                    y_data.push(label);
                }
            }
        }
    }

    println!("Dataset chargé : {} images", x_data.len());
    (x_data, y_data)
}
