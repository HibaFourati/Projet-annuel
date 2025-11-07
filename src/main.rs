mod data;
mod linear_model;

use data::load_dataset;
use linear_model::{LinearModel, train, evaluate_model};

fn main() {
    let (x_data, y_data) = load_dataset("dataset");

    println!("Nombre total d'images : {}", x_data.len());
    println!("Nombre total de labels : {}", y_data.len());
    println!("Taille d'une image vectorisée : {}", x_data[0].len());

    // Créer le modèle linéaire
    let n_features = x_data[0].len();
    let n_classes = 3;
    let mut model = LinearModel::new(n_features, n_classes);

    // Entraîner le modèle
    let epochs = 10;
    let lr = 0.01;
    println!("Début de l'entraînement...");
    train(&mut model, &x_data, &y_data, epochs, lr);
    println!("Entraînement terminé !");

    // Tester sur la première image
    let pred = model.forward(&x_data[0]);
    println!("Prediction de la première image : {:?}", pred);
    println!("Label réel : {}", y_data[0]);

    // Évaluer le modèle sur tout le dataset
    evaluate_model(&model, &x_data, &y_data);
}

