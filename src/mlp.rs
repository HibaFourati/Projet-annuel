// src/mlp.rs

use rand::Rng;

pub struct MLP {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub learning_rate: f64,
    pub weights_input_hidden: Vec<Vec<f64>>,
    pub weights_hidden_output: Vec<Vec<f64>>,
    pub bias_hidden: Vec<f64>,
    pub bias_output: Vec<f64>,
}

impl MLP {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();

        let weights_input_hidden = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let weights_hidden_output = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let bias_hidden = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_output = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        MLP {
            input_size,
            hidden_size,
            output_size,
            learning_rate,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn dsigmoid(y: f64) -> f64 {
        y * (1.0 - y)
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        let hidden: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                let sum: f64 = self.weights_input_hidden[i]
                    .iter()
                    .zip(inputs)
                    .map(|(w, inp)| w * inp)
                    .sum::<f64>()
                    + self.bias_hidden[i];
                Self::sigmoid(sum)
            })
            .collect();

        let outputs: Vec<f64> = (0..self.output_size)
            .map(|i| {
                let sum: f64 = self.weights_hidden_output[i]
                    .iter()
                    .zip(&hidden)
                    .map(|(w, h)| w * h)
                    .sum::<f64>()
                    + self.bias_output[i];
                Self::sigmoid(sum)
            })
            .collect();

        (hidden, outputs)
    }

    pub fn train(&mut self, inputs: &Vec<f64>, targets: &Vec<f64>) {
        // Forward pass
        let (hidden, outputs) = self.forward(inputs);

        // Calcul des erreurs de sortie
        let output_errors: Vec<f64> = targets
            .iter()
            .zip(outputs.iter())
            .map(|(t, o)| t - o)
            .collect();

        // Gradient de sortie
        let output_gradients: Vec<f64> = outputs
            .iter()
            .zip(output_errors.iter())
            .map(|(o, e)| Self::dsigmoid(*o) * e * self.learning_rate)
            .collect();

        // Ajustement des poids -> output
        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                self.weights_hidden_output[i][j] += output_gradients[i] * hidden[j];
            }
            self.bias_output[i] += output_gradients[i];
        }

        // Erreurs cachées
        let hidden_errors: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                (0..self.output_size)
                    .map(|j| self.weights_hidden_output[j][i] * output_errors[j])
                    .sum()
            })
            .collect();

        // Gradient caché
        let hidden_gradients: Vec<f64> = hidden
            .iter()
            .zip(hidden_errors.iter())
            .map(|(h, e)| Self::dsigmoid(*h) * e * self.learning_rate)
            .collect();

        // Ajustement des poids input -> hidden
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                self.weights_input_hidden[i][j] += hidden_gradients[i] * inputs[j];
            }
            self.bias_hidden[i] += hidden_gradients[i];
        }
    }
}
