use arff_reader::data_set::DataSet;
use std::{iter, time::Instant};

// This takes 30 seconds to run on release build.
// USE RELEASE FLAG FOR THIS (will take forever otherwise!)
// This is a late night code spike, will clean up soon!
fn main() {
    let now = Instant::now();

    let target_index = 3;
    let train_set = DataSet::import("./test_data/chapter_7_numeric.arff").expect("File not found");
    let mut linear_model = LinearModel::new(&train_set, target_index);
    linear_model.regress();

    println!("Finished in {:?}", now.elapsed())
}

struct LinearModel {
    data: Vec<Vec<f64>>,
    max_values: Vec<f64>,
    weights: Vec<f64>,
    attribute_map: Vec<usize>,
    target_index: usize,
    predictions: Vec<f64>,
    sum_of_squared_error: f64,
}

impl LinearModel {
    fn new(train_set: &DataSet, target_index: usize) -> Self {
        let mut data: Vec<_> = (0..(train_set.get_len()))
            .map(|index| train_set.get_attributes()[index].assume_numeric())
            .map(|array| array.iter().map(|&x| x as f64).collect::<Vec<_>>())
            .collect();

        let max_values: Vec<_> = data
            .iter()
            .map(|array| {
                array
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .clone()
            })
            .collect();

        // Normalize data
        data = data
            .iter()
            .enumerate()
            .map(|(index, array)| array.iter().map(|x| x / max_values[index]).collect())
            .collect();

        let attribute_map: Vec<_> = (0..(train_set.get_len()))
            .filter(|index| *index != target_index)
            .collect();

        // Last index is the bias term
        let weights = vec![0.1; attribute_map.len() + 1];
        let predictions = vec![0.0; data[0].len()];
        let sum_of_squared_error = f64::MAX;

        Self {
            data,
            max_values,
            weights,
            attribute_map,
            target_index,
            predictions,
            sum_of_squared_error,
        }
    }

    fn regress(&mut self) {
        let batch_size = 100;
        let mut last_error = f64::MAX;
        let mut iterations = 0;

        for batch in 0.. {
            println!("Batch: {}", batch);
            for _ in 0..batch_size {
                iterations += 1;

                let attribute_error_deltas: Vec<_> = self
                    .attribute_map
                    .iter()
                    .map(|&index| {
                        (0..self.data[0].len())
                            .map(|instance| {
                                (
                                    self.data[self.target_index][instance],
                                    self.predictions[instance],
                                    self.data[index][instance],
                                )
                            })
                            .map(|(actual, predicted, attribute_value)| {
                                (actual - predicted) * attribute_value
                            })
                            .sum::<f64>()
                    })
                    // Bias term
                    .chain(iter::once(
                        (0..self.data[0].len())
                            .map(|instance| {
                                (
                                    self.data[self.target_index][instance],
                                    self.predictions[instance],
                                )
                            })
                            .map(|(actual, predicted)| (actual - predicted))
                            .sum::<f64>(),
                    ))
                    .collect();

                let old_weights = self.weights.clone();
                let mut alpha_low = 0.000_000_001;
                let mut alpha_high = 1.0;

                for _ in 0..5 {
                    (0..self.weights.len()).for_each(|index| {
                        self.weights[index] =
                            old_weights[index] + attribute_error_deltas[index] * alpha_low
                    });
                    let error_low = self.sum_of_squared_error();

                    (0..self.weights.len()).for_each(|index| {
                        self.weights[index] =
                            old_weights[index] + attribute_error_deltas[index] * alpha_high
                    });
                    let error_high = self.sum_of_squared_error();
                    // This will be the sum of squared error if loop breaks
                    self.sum_of_squared_error = error_high;

                    let alpha = (alpha_high + alpha_low) * 0.5;

                    if error_low < error_high {
                        alpha_high = alpha
                    } else {
                        alpha_low = alpha
                    }
                }
            }

            let error_delta = (last_error - self.sum_of_squared_error).abs();
            println!("{}", error_delta);
            if error_delta < f64::MIN_POSITIVE {
                break;
            }
            last_error = self.sum_of_squared_error;
        }

        // Did some algebra to work out the converted values
        let converted_weights: Vec<_> = (0..self.weights.len() - 1)
            .map(|index| {
                (self.max_values.last().unwrap() / self.max_values[index]) * self.weights[index]
            })
            .chain(iter::once(
                self.weights.last().unwrap() * self.max_values.last().unwrap(),
            ))
            .collect();

        let converted_sum_of_squared_error =
            self.sum_of_squared_error * self.max_values.last().unwrap().powi(2);

        println!("{}", iterations);
        println!("{:?}", converted_weights);
        println!("{}", converted_sum_of_squared_error);
    }

    fn sum_of_squared_error(&mut self) -> f64 {
        self.predictions = (0..self.predictions.len())
            .map(|instance| {
                // bias
                self.weights.last().unwrap()
        // dot product
            + (0..self.attribute_map.len())
                .map(|index| self.weights[index] * self.data[self.attribute_map[index]][instance])
                .sum::<f64>()
            })
            .collect();

        (0..self.predictions.len())
            .map(|instance| {
                (
                    self.data[self.target_index][instance],
                    self.predictions[instance],
                )
            })
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f64>()
    }
}
