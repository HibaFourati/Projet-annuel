
/*mod mlp;
mod utils;

// importer le fichier mlp.rs
use mlp::MLP;     // importe la struct publique MLP

use std::fs::File;
use std::io::Write;

// Fonction pour entraîner un test et générer CSV
fn train_test(inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, file_name: &str, test_name: &str) {
    let mut model = MLP::new(inputs[0].len(), 3, targets[0].len(), 0.5); // 3 neurones cachés, lr=0.5

    let mut file = File::create(file_name).expect("Erreur création du fichier CSV");
    writeln!(file, "epoch,loss").unwrap();

    for epoch in 0..10000 {
        let mut total_loss = 0.0;
        for i in 0..inputs.len() {
            model.train(&inputs[i], &targets[i]);

            let (_, output) = model.forward(&inputs[i]);
            let loss = (targets[i][0] - output[0]).powi(2);
            total_loss += loss;
        }

        if epoch % 100 == 0 {
            println!("{} - Epoch {} - Loss: {:.6}", test_name, epoch, total_loss);
            writeln!(file, "{},{}", epoch, total_loss).unwrap();
        }
    }

    println!("\n{} - Résultats finaux :", test_name);
    for i in 0..inputs.len() {
        let (_, output) = model.forward(&inputs[i]);
        println!("{:?} -> {:?}", inputs[i], output);
    }
}

fn main() {
    // Test XOR
    let xor_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let xor_targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];
    train_test(xor_inputs, xor_targets, "loss_xor.csv", "XOR");

    // Test linéaire simple
    let linear_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let linear_targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
    ];
    train_test(linear_inputs, linear_targets, "loss_linear.csv", "Linéaire");
}
*/
/*mod mlp;
use mlp::MLP;

use std::fs::File;
use std::io::{BufReader, BufRead, Write};

// Fonction pour lire le CSV
fn read_csv(path: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let file = File::open(path).expect("Impossible d'ouvrir le CSV");
    let reader = BufReader::new(file);

    let mut inputs = vec![];
    let mut targets = vec![];

    for line in reader.lines() {
        let line = line.unwrap();
        let numbers: Vec<f64> = line
            .split(',')
            .map(|x| x.parse::<f64>().unwrap())
            .collect();

        let input = numbers[..784].to_vec();   // 28x28 pixels = 784
        let target = numbers[784..].to_vec();  // 3 classes : piano/guitare/violon
        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}

fn main() {
    // Lire le dataset
    let (inputs, targets) = read_csv("dataset_instruments.csv");

    // Créer le PMC
    let input_size = 784;   // 28*28 pixels
    let hidden_size = 64;
    let output_size = 3;    // 3 classes
    let learning_rate = 0.1;

    let mut model = MLP::new(input_size, hidden_size, output_size, learning_rate);

    // Préparer le fichier pour la courbe de perte
    let mut file = File::create("loss_dataset.csv").expect("Impossible de créer le fichier CSV");
    writeln!(file, "epoch,loss").unwrap();

    // Entraînement
    let epochs = 1000;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for i in 0..inputs.len() {
            model.train(&inputs[i], &targets[i]);
            let (_, output) = model.forward(&inputs[i]);
            let loss: f64 = targets[i].iter()
                .zip(output.iter())
                .map(|(t,o)| (t - o).powi(2))
                .sum();
            total_loss += loss;
        }

        if epoch % 50 == 0 {
            println!("Epoch {} - Loss: {:.6}", epoch, total_loss);
            writeln!(file, "{},{}", epoch, total_loss).unwrap();
        }
    }

    // Résultats finaux
    for i in 0..inputs.len() {
        let (_, output) = model.forward(&inputs[i]);
        println!("Input {} -> {:?}", i, output);
    }
}
*/
mod mlp;
use mlp::MLP;

use std::fs::File;
use std::io::{BufReader, BufRead, Write};

// Fonction pour lire le CSV
fn read_csv(path: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let file = File::open(path).expect("Impossible d'ouvrir le CSV");
    let reader = BufReader::new(file);

    let mut inputs = vec![];
    let mut targets = vec![];

    for line in reader.lines() {
        let line = line.unwrap();
        let numbers: Vec<f64> = line
            .split(',')
            .map(|x| x.parse::<f64>().unwrap())
            .collect();

        let input = numbers[..784].to_vec();   // 28x28 pixels = 784
        let target = numbers[784..].to_vec();  // 3 classes : piano/guitare/violon
        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}

// Fonction pour obtenir l'indice du maximum d'un vecteur
fn argmax(vec: &Vec<f64>) -> usize {
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0
}

fn main() {
    // Lire le dataset
    let (inputs, targets) = read_csv("dataset_instruments.csv");

    // Créer le PMC
    let input_size = 784;   // 28*28 pixels
    let hidden_size = 64;
    let output_size = 3;    // 3 classes
    let learning_rate = 0.1;

    let mut model = MLP::new(input_size, hidden_size, output_size, learning_rate);

    // Préparer le fichier pour la courbe de perte
    let mut file = File::create("loss_dataset.csv").expect("Impossible de créer le fichier CSV");
    writeln!(file, "epoch,loss").unwrap();

    // Entraînement
    let epochs = 1000;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for i in 0..inputs.len() {
            model.train(&inputs[i], &targets[i]);
            let (_, output) = model.forward(&inputs[i]);
            let loss: f64 = targets[i].iter()
                .zip(output.iter())
                .map(|(t,o)| (t - o).powi(2))
                .sum();
            total_loss += loss;
        }

        // Écrire chaque epoch pour une courbe fluide
        writeln!(file, "{},{}", epoch, total_loss).unwrap();

        if epoch % 50 == 0 {
            println!("Epoch {} - Loss: {:.6}", epoch, total_loss);
        }
    }

    // Calcul de la précision finale
    let mut correct = 0;
    for i in 0..inputs.len() {
        let (_, output) = model.forward(&inputs[i]);
        let predicted_class = argmax(&output);
        let actual_class = argmax(&targets[i]);
        if predicted_class == actual_class {
            correct += 1;
        }
    }
    let accuracy = (correct as f64 / inputs.len() as f64) * 100.0;
    println!("\nPrécision finale sur le dataset : {:.2}%", accuracy);

    // Résultats finaux (quelques exemples)
    println!("\nRésultats finaux pour les 10 premiers exemples :");
    for i in 0..inputs.len().min(10) {
        let (_, output) = model.forward(&inputs[i]);
        println!("Input {} -> {:?}", i, output);
    }
}

