use image::{io::Reader as ImageReader, GenericImageView};
use std::error::Error;
use std::fs;
use std::path::Path;

pub struct DatasetLoader;

impl DatasetLoader {
    pub fn load_image_dataset(dataset_path: &str, image_size: (u32, u32)) -> Result<(Vec<Vec<f64>>, Vec<usize>), Box<dyn Error>> {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        let classes = ["guitare", "piano", "violon"];
        
        for (class_idx, class_name) in classes.iter().enumerate() {
            let class_path = Path::new(dataset_path).join(class_name);
            
            println!(" Chargement des images de : {}", class_name);
            
            let entries = fs::read_dir(&class_path)?;
            let mut image_count = 0;
            
            for entry in entries {
                let entry = entry?;
                let path = entry.path();
                
                if path.extension().map_or(false, |ext| {
                    ext == "jpg" || ext == "jpeg" || ext == "png"
                }) {
                    match Self::load_and_process_image(&path, image_size) {
                        Ok(image_features) => {
                            features.push(image_features);
                            labels.push(class_idx);
                            image_count += 1;
                        }
                        Err(e) => println!("❌ Erreur sur {}: {}", path.display(), e),
                    }
                }
            }
            
            println!(" {} images chargées pour {}", image_count, class_name);
        }
        
        println!(" Dataset chargé : {} images, {} classes", features.len(), classes.len());
        Ok((features, labels))
    }
    
    fn load_and_process_image(path: &Path, target_size: (u32, u32)) -> Result<Vec<f64>, Box<dyn Error>> {
        let img = ImageReader::open(path)?.decode()?;
        let resized = img.resize_exact(target_size.0, target_size.1, image::imageops::FilterType::Lanczos3);
        
        let mut features = Vec::new();
        
        // Conversion en niveaux de gris et normalisation
        for pixel in resized.to_luma8().pixels() {
            let gray = pixel[0] as f64 / 255.0;
            features.push(gray);
        }
        
        Ok(features)
    }
    
    pub fn get_class_names() -> Vec<&'static str> {
        vec!["guitare", "piano", "violon"]
    }
}