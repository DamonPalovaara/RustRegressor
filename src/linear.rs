use std::iter;

use crate::data_set::DataSet;
use itertools::Itertools;
use rand::{seq::SliceRandom, thread_rng};

const CHUNK_SIZE: usize = 1024; // Size of sub-slice of training data for each batch
const BATCH_SIZE: usize = 100; // Number of adjustments for each chunk
const PATIENCE: usize = 1000; // Quit after this many batches with no progress
const ITERATIONS: usize = 30; // 20 to 40 seems like a good range
const ALPHA_LOW: f64 = 0.000_000_000_1;
const ALPHA_MAX: f64 = 1.0;

pub struct LinearModel {
    data: Vec<Vec<f64>>,
    max_values: Vec<f64>,
    min_values: Vec<f64>,
    weights: Vec<f64>,
    attribute_map: Vec<usize>,
    target_index: usize,
    predictions: Vec<f64>,
    partial_predictions: Vec<f64>,
    sum_of_squared_error: f64,
}

impl LinearModel {
    pub fn new(train_set: &DataSet, target_index: usize) -> Self {
        let mut data: Vec<_> = (0..(train_set.get_len()))
            .map(|index| train_set.get_attributes()[index].assume_numeric())
            .map(|array| array.iter().map(|&x| x as f64).collect::<Vec<_>>())
            .collect();

        let one_over_n = 1.0 / data[0].len() as f64;
        let (min_values, max_values, averages): (Vec<_>, Vec<_>, Vec<_>) = data
            .iter()
            .map(|array| {
                array
                    .iter()
                    .map(|x| (*x, *x, 0.0))
                    .reduce(|(min, max, mut average), (value, _, _)| {
                        average += value.ln();
                        if value < min {
                            (value, max, average)
                        } else if value > max {
                            (min, value, average)
                        } else {
                            (min, max, average)
                        }
                    })
                    .map(|(min, max, average)| (min, max, f64::exp(average * one_over_n)))
                    .unwrap()
            })
            .multiunzip();

        println!("{:?}", min_values);
        println!("{:?}", max_values);
        println!("{:?}", averages);

        // Normalize data
        data = data
            .iter()
            .enumerate()
            .map(|(index, array)| {
                array
                    .iter()
                    .map(|x| (x - min_values[index]) / (max_values[index] - min_values[index]))
                    .collect()
            })
            .collect();

        let attribute_map: Vec<_> = (0..(train_set.get_len()))
            .filter(|index| *index != target_index)
            .collect();

        // Last index is the bias term
        let weights = vec![0.1; attribute_map.len() + 1];
        // let weights = vec![0.33323, 0.33324, 0.33099, 0.00286, -0.00014];
        let predictions = vec![0.0; data[0].len()];
        let partial_predictions = vec![0.0; data[0].len().min(CHUNK_SIZE)];
        let sum_of_squared_error = f64::MAX;

        Self {
            data,
            max_values,
            min_values,
            weights,
            attribute_map,
            target_index,
            predictions,
            partial_predictions,
            sum_of_squared_error,
        }
    }

    pub fn regress(&mut self) {
        let mut smallest_error = f64::MAX;
        let mut iterations = 0;
        let mut batches_without_progress = 0;
        let mut random_indices: Vec<_> = (0..self.data[0].len()).collect();
        let mut best_weights = self.weights.clone();

        let mut rng = thread_rng();
        random_indices.shuffle(&mut rng);
        let binding = random_indices
            .into_iter()
            .cycle()
            .chunks(self.partial_predictions.len());
        let mut random_index_iter = binding.into_iter();

        let mut random_indices: Vec<_> = random_index_iter.next().unwrap().collect();

        for batch in 0.. {
            println!("Batch: {}", batch);
            for _ in 0..BATCH_SIZE {
                iterations += 1;

                let attribute_error_deltas: Vec<_> = self
                    .attribute_map
                    .iter()
                    .map(|&index| {
                        random_indices
                            .iter()
                            .map(|&instance| {
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
                        random_indices
                            .iter()
                            .map(|&instance| {
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
                let mut alpha_low = ALPHA_LOW;
                let mut alpha_high = ALPHA_MAX;
                for _ in 0..ITERATIONS {
                    (0..self.weights.len()).for_each(|index| {
                        self.weights[index] =
                            old_weights[index] + attribute_error_deltas[index] * alpha_low
                    });
                    let error_low = self.partial_sum_of_squared_error(&random_indices);

                    (0..self.weights.len()).for_each(|index| {
                        self.weights[index] =
                            old_weights[index] + attribute_error_deltas[index] * alpha_high
                    });
                    let error_high = self.partial_sum_of_squared_error(&random_indices);

                    let alpha = (alpha_high + alpha_low) * 0.5;

                    // println!("{:.2} \t {:.2} \t {:.2}", alpha, error_low, error_high);

                    if error_low < error_high {
                        alpha_high = alpha
                    } else {
                        alpha_low = alpha
                    }
                }
                random_indices = random_index_iter.next().unwrap().collect();
            }

            self.sum_of_squared_error = self.sum_of_squared_error();
            println!("SSE: {}", self.sum_of_squared_error);
            println!("{}", batches_without_progress);

            if smallest_error <= self.sum_of_squared_error {
                batches_without_progress += 1;
                if batches_without_progress > PATIENCE {
                    self.weights = best_weights;
                    self.sum_of_squared_error = self.sum_of_squared_error();
                    break;
                }
            } else {
                smallest_error = self.sum_of_squared_error;
                best_weights = self.weights.clone();
                batches_without_progress = 0;
            }
        }

        // Did some tricky algebra to convert back to un-normalized weights
        let converted_weights: Vec<_> = (0..self.weights.len() - 1)
            .map(|index| {
                ((self.max_values.last().unwrap() - self.min_values.last().unwrap())
                    / (self.max_values[index] - self.min_values[index]))
                    * self.weights[index]
            })
            .chain(iter::once(
                (0..self.weights.len() - 1)
                    .map(|index| {
                        ((self.min_values[index] * self.weights[index])
                            * (self.min_values.last().unwrap() - self.max_values.last().unwrap()))
                            / (self.max_values[index] - self.min_values[index])
                    })
                    .sum::<f64>()
                    + self.min_values.last().unwrap()
                    + (self.weights.last().unwrap()
                        * (self.max_values.last().unwrap() - self.min_values.last().unwrap())),
            ))
            .collect();

        let converted_sum_of_squared_error = self.sum_of_squared_error
            * (self.max_values.last().unwrap() - self.min_values.last().unwrap()).powi(2);

        println!("Iterations: {}", iterations);
        println!("Raw weights: {:.5?}", self.weights);
        println!("Weights: {:.5?}", converted_weights);
        println!("Raw SSE: {}", self.sum_of_squared_error);
        println!("SSE: {}", converted_sum_of_squared_error);
    }

    fn sum_of_squared_error(&mut self) -> f64 {
        self.predictions = (0..self.predictions.len())
            .map(|instance| {
                self.weights.last().unwrap()
                    + (0..self.attribute_map.len())
                        .map(|index| {
                            self.weights[index] * self.data[self.attribute_map[index]][instance]
                        })
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

    fn partial_sum_of_squared_error(&mut self, random_indices: &[usize]) -> f64 {
        self.partial_predictions = random_indices
            .iter()
            .map(|&instance| {
                self.weights.last().unwrap()
                    + (0..self.attribute_map.len())
                        .map(|index| {
                            self.weights[index] * self.data[self.attribute_map[index]][instance]
                        })
                        .sum::<f64>()
            })
            .collect();

        (0..self.partial_predictions.len())
            .map(|index| {
                (
                    self.data[self.target_index][random_indices[index]],
                    self.partial_predictions[index],
                )
            })
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f64>()
    }
}
