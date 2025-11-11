use std::fs::File;
use std::io::{BufReader, BufRead};

fn read_csv(path: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let file = File::open(path).expect("Impossible d'ouvrir le CSV");
    let reader = BufReader::new(file);

    let mut inputs = vec![];
    let mut targets = vec![];

    for line in reader.lines() {
        let line = line.unwrap();
        let numbers: Vec<f64> = line.split(',')
            .map(|x| x.parse::<f64>().unwrap())
            .collect();
        let input = numbers[..784].to_vec();   // 28x28 pixels = 784
        let target = numbers[784..].to_vec();  // 3 classes : piano/guitare/violon
        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}
