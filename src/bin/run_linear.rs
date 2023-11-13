use arff_reader::data_set::DataSet;
use std::time::Instant;

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
            .collect();

        let attributes: Vec<_> = (0..(train_set.get_len()))
            .filter(|index| *index != target_index)
            .collect();

        // Last index is the bias term
        let mut weights = vec![0.1; attributes.len() + 1];
        let bias_index = weights.len() - 1;

        let mut predictions = vec![0.0; data[0].len()];

        let rate = 0.000_000_2;
        let mut bias_error_delta = 0.0;
        let mut sum_squared_error = 0.0;

        for _ in 0..100_000_000 {
            predictions = (0..predictions.len())
                .map(|instance| {
                    // bias
                    weights[bias_index]
                    // dot product
                        + (0..attributes.len())
                            .map(|index| weights[index] * data[attributes[index]][instance])
                            .sum::<f32>()
                })
                .collect();

            sum_squared_error = (0..predictions.len())
                .map(|instance| (data[target_index][instance], predictions[instance]))
                .map(|(actual, predicted)| (actual - predicted).powi(2))
                .sum::<f32>();

            bias_error_delta = (0..data[0].len())
                .map(|instance| (data[target_index][instance], predictions[instance]))
                .map(|(actual, predicted)| (actual - predicted))
                .sum::<f32>();

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
                        .sum::<f32>()
                })
                .collect();

            // Update bias
            weights[bias_index] += bias_error_delta * rate;
            (0..(weights.len() - 1))
                .for_each(|index| weights[index] += attribute_error_deltas[index] * rate);
        }

        println!("{:?}", weights);
        println!("{}", sum_squared_error);

        Self {}
    }

    fn test(&self, test_set: &DataSet, target_index: usize) {}
}

fn error_delta(actual: f32, predicted: f32, attribute_value: f32) -> f32 {
    (actual - predicted) * attribute_value
}
