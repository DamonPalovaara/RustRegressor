use arff_reader::data_set::DataSet;
use std::{error, iter, time::Instant};

// This takes 30 seconds to run on release build.
// USE RELEASE FLAG FOR THIS (will take forever otherwise!)
// This is a late night code spike, will clean up soon!
fn main() {
    let now = Instant::now();

    let target_index = 3;
    let train_set = DataSet::import("./test_data/chapter_7_numeric.arff").expect("File not found");
    let linear_model = LinearModel::new(&train_set, target_index);

    println!("Finished in {:?}", now.elapsed())
}

struct LinearModel {}

impl LinearModel {
    fn new(train_set: &DataSet, target_index: usize) -> Self {
        let data: Vec<_> = (0..(train_set.get_len()))
            .map(|index| train_set.get_attributes()[index].assume_numeric())
            .map(|array| array.iter().map(|&x| x as f64).collect::<Vec<_>>())
            .collect();

        let attributes: Vec<_> = (0..(train_set.get_len()))
            .filter(|index| *index != target_index)
            .collect();

        // Last index is the bias term
        let mut weights = vec![0.1; attributes.len() + 1];
        let bias_index = weights.len() - 1;

        let mut predictions = vec![0.0; data[0].len()];

        let batch_size = 10_000_000;
        let rate = 0.000_000_2;
        let mut sum_squared_error = 0.0;
        let mut last_sum_squared_error = 0.0;
        let threshold = f64::MIN_POSITIVE;
        let mut iterations = 0;

        for batch in 0.. {
            println!("Batch: {}", batch);
            for _ in 0..batch_size {
                iterations += 1;

                predictions = (0..predictions.len())
                    .map(|instance| {
                        // bias
                        weights[bias_index]
                    // dot product
                        + (0..attributes.len())
                            .map(|index| weights[index] * data[attributes[index]][instance])
                            .sum::<f64>()
                    })
                    .collect();

                sum_squared_error = (0..predictions.len())
                    .map(|instance| (data[target_index][instance], predictions[instance]))
                    .map(|(actual, predicted)| (actual - predicted).powi(2))
                    .sum::<f64>();

                let bias_error_delta = (0..data[0].len())
                    .map(|instance| (data[target_index][instance], predictions[instance]))
                    .map(|(actual, predicted)| (actual - predicted))
                    .sum::<f64>();

                let attribute_error_deltas: Vec<_> = attributes
                    .iter()
                    .map(|&index| {
                        (0..data[0].len())
                            .map(|instance| {
                                (
                                    data[target_index][instance],
                                    predictions[instance],
                                    data[index][instance],
                                )
                            })
                            .map(|(actual, predicted, attribute_value)| {
                                (actual - predicted) * attribute_value
                            })
                            .sum::<f64>()
                    })
                    .collect();

                // Update bias
                weights[bias_index] += bias_error_delta * rate;
                (0..(weights.len() - 1))
                    .for_each(|index| weights[index] += attribute_error_deltas[index] * rate);
            }
            let error_delta = (last_sum_squared_error - sum_squared_error).abs();
            println!("{:.20}", error_delta);
            if error_delta <= threshold {
                break;
            }
            last_sum_squared_error = sum_squared_error;
        }

        println!("{}", iterations);
        println!("{:?}", weights);
        println!("{}", sum_squared_error);

        Self {}
    }

    fn test(&self, test_set: &DataSet, target_index: usize) {}
}
