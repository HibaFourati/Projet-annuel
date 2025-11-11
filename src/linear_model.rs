use rand::Rng;

#[derive(Debug)]
pub struct LinearModel {
    pub weights: Vec<Vec<f64>>, // n_features x n_classes
    pub bias: Vec<f64>,         // n_classes
}

impl LinearModel {
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..n_features)
            .map(|_| (0..n_classes).map(|_| rng.gen::<f64>() * 0.01).collect())
            .collect();
        let bias = vec![0.0; n_classes];
        LinearModel { weights, bias }
    }

    pub fn forward(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut logits = vec![0.0; self.bias.len()];
        for i in 0..self.weights.len() {
            for j in 0..self.bias.len() {
                logits[j] += self.weights[i][j] * x[i];
            }
        }
        for j in 0..logits.len() {
            logits[j] += self.bias[j];
        }
        softmax(&logits)
    }

    pub fn predict(&self, x: &Vec<f64>) -> Vec<f64> {
        self.forward(x)
    }
}

pub fn softmax(logits: &Vec<f64>) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

pub fn cross_entropy_loss(pred: &Vec<f64>, label: usize) -> f64 {
    -pred[label].ln()
}

pub fn update_model(model: &mut LinearModel, x: &Vec<f64>, y: usize, lr: f64) {
    let pred = model.forward(x);
    let mut grad = vec![0.0; pred.len()];
    for j in 0..pred.len() {
        grad[j] = pred[j] - if j == y { 1.0 } else { 0.0 };
    }
    for i in 0..model.weights.len() {
        for j in 0..model.bias.len() {
            model.weights[i][j] -= lr * grad[j] * x[i];
        }
    }
    for j in 0..model.bias.len() {
        model.bias[j] -= lr * grad[j];
    }
}

pub fn train(model: &mut LinearModel, x_data: &Vec<Vec<f64>>, y_data: &Vec<usize>, epochs: usize, lr: f64) {
    for _ in 0..epochs {
        for (x, &y) in x_data.iter().zip(y_data.iter()) {
            update_model(model, x, y, lr);
        }
    }
}

pub fn evaluate_model(model: &LinearModel, x_data: &Vec<Vec<f64>>, y_data: &Vec<usize>) {
    let class_names = vec!["guitare", "piano", "violon"];
    let mut correct = 0;
    let mut class_correct = vec![0; class_names.len()];
    let mut class_total = vec![0; class_names.len()];

    for (x, &y_true) in x_data.iter().zip(y_data.iter()) {
        let pred = model.predict(x);
        let predicted_label = pred
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if predicted_label == y_true {
            correct += 1;
            class_correct[y_true] += 1;
        }
        class_total[y_true] += 1;
    }

    let accuracy = correct as f64 / x_data.len() as f64 * 100.0;
    println!("Précision globale : {:.2}%", accuracy);

    for i in 0..class_names.len() {
        let acc = if class_total[i] > 0 {
            class_correct[i] as f64 / class_total[i] as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "Classe {} : {:.2}% ({} / {})",
            class_names[i], acc, class_correct[i], class_total[i]
        );
    }
}
