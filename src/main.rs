mod linear_model;
mod image_features;

use linear_model::{LinearModel, train, evaluate_model};
use image_features::DatasetLoader;

// Fonction de normalisation à ajouter
fn normalize_dataset(x_data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if x_data.is_empty() {
        return Vec::new();
    }
    let n_features = x_data[0].len();
    let mut normalized = x_data.to_vec();
    for feature_idx in 0..n_features {
        let values: Vec<f64> = x_data.iter().map(|x| x[feature_idx]).collect();
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let std: f64 = (values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        for i in 0..normalized.len() {
            if std > 1e-10 {
                normalized[i][feature_idx] = (normalized[i][feature_idx] - mean) / std;
            } else {
                normalized[i][feature_idx] = normalized[i][feature_idx] - mean;
            }
        }
    }
    normalized
}

// Structure pour sauvegarder les résultats (ajoute-la)
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

#[derive(Serialize, Deserialize, Clone)]
struct TrainingResults {
    loss_history: Vec<f64>,
    accuracy_history: Vec<f64>,
    predictions: Vec<usize>,
    true_labels: Vec<usize>,
}

impl TrainingResults {
    fn save_to_json(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        println!(" Résultats sauvegardés dans {}", filename);
        Ok(())
    }
}

// Version modifiée de train() qui retourne des résultats
fn train_with_results(model: &mut LinearModel, x_data: &[Vec<f64>], y_data: &[usize], epochs: usize, lr: f64) -> TrainingResults {
    let mut results = TrainingResults {
        loss_history: Vec::new(),
        accuracy_history: Vec::new(),
        predictions: Vec::new(),
        true_labels: y_data.to_vec(),
    };

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (x, &y) in x_data.iter().zip(y_data.iter()) {
            let pred = model.forward(x);
            let loss = linear_model::cross_entropy_loss(&pred, y);
            total_loss += loss;

            let predicted_label = pred.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted_label == y {
                correct += 1;
            }

            linear_model::update_model(model, x, y, lr);
        }

        let avg_loss = total_loss / x_data.len() as f64;
        let accuracy = correct as f64 / x_data.len() as f64;

        results.loss_history.push(avg_loss);
        results.accuracy_history.push(accuracy);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("Epoch {}/{} - Loss: {:.4}, Accuracy: {:.4}", 
                    epoch + 1, epochs, avg_loss, accuracy);
        }
    }

    // Prédictions finales
    for x in x_data {
        let pred = model.predict(x);
        let predicted_label = pred.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        results.predictions.push(predicted_label);
    }

    results
}

fn main() {
    println!(" CHARGEMENT DES DONNÉES RÉELLES");
    
    // 1. Charge tes VRAIES données
    let dataset_path = "dataset";
    let image_size = (32, 32);
    
    let (x_data, y_data) = DatasetLoader::load_image_dataset(dataset_path, image_size)
        .expect(" Erreur chargement données");
    
    println!(" Données chargées : {} images, {} features", x_data.len(), x_data[0].len());
    
    // 2. Normalise les données
    let x_normalized = normalize_dataset(&x_data);
    
    // 3. Crée le modèle pour 3 classes
    let n_features = x_normalized[0].len();
    let n_classes = 3;
    
    let mut model = LinearModel::new(n_features, n_classes);
    
    // 4. Entraîne le modèle avec tracking des résultats
    let learning_rate = 0.01;
    let epochs = 50;
    
    println!(" Démarrage de l'entraînement...");
    let results = train_with_results(&mut model, &x_normalized, &y_data, epochs, learning_rate);
    
    // 5. Évaluation finale
    println!("\nÉVALUATION FINALE:");
    evaluate_model(&model, &x_normalized, &y_data);
    
    // 6. Sauvegarde les résultats pour les graphiques
    results.save_to_json("training_results.json").unwrap();
    
    // 7. Affichage des courbes
    println!("\n COURBES D'APPRENTISSAGE:");
    println!("Loss initiale: {:.4}", results.loss_history[0]);
    println!("Loss finale: {:.4}", results.loss_history.last().unwrap());
    println!("Accuracy initiale: {:.4}", results.accuracy_history[0]);
    println!("Accuracy finale: {:.4}", results.accuracy_history.last().unwrap());
    
    println!("\n PRÊT POUR LES GRAPHIQUES !");
    println!(" Fichier des résultats: training_results.json");
}