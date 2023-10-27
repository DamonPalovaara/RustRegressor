use std::collections::BTreeMap;

use crate::{data_set::DataSet, test_statistics::ConfusionMatrix};

/// Basic implementation of the Naive Bayes algorithm
/// with smoothing
pub struct NaiveBayes {
    // array[target_value] = P(target_value)
    probability_target: Vec<f32>,
    // array[target_value][attribute_index][attribute_value] = P(target_value | attribute_value)
    probability_given: Vec<Vec<Vec<f32>>>,
}

impl NaiveBayes {
    /// Creates model trained on data
    /// Assumes all data is nominal, will panic if any attribute is numeric
    /// Use new_from_numeric(...) to handle importing numeric data
    /// k is the smoothing factor
    pub fn new(data: &DataSet, target: usize, k: usize) -> Self {
        // Target value -> [indices with target value]
        // btw I should probably mention what '_' are:
        //  In this context they tell compiler to deduce the type
        //  Type inference is usually pretty good but there are areas that it struggles
        // let mut target_indices = BTreeMap::<_, Vec<usize>>::new();
        let target_indices = (0..data.get_data_len())
            .map(|index| (index, data.get_value(target, index).assume_nominal()))
            .fold(
                BTreeMap::<_, Vec<_>>::new(),
                |mut indices, (index, target)| {
                    indices
                        .entry(target)
                        .and_modify(|entry| entry.push(index))
                        .or_insert(vec![index]);
                    indices
                },
            );

        // target value -> probability of target value
        // You need to tell Rust the type of collection to collect into
        // However it's pretty simple to infer the type inside the collection
        // An alternative to Vec<_> is .collect::<Vec<_>>()
        let probability_target: Vec<_> = target_indices
            .values()
            .map(|indices| indices.len() as f32 / data.get_data_len() as f32)
            .collect(); // .collect::<Vec<_>>();

        // attribute indices to iterate over
        let attribute_indices: Vec<_> = (0..data.get_len())
            .filter(|&index| index != target)
            .collect();

        // array[target_value][attribute_index][attribute_value] = P(target_value | attribute_value)
        // This is nasty, I'm sorry. I probably should just be using for loops here
        let probability_given: Vec<_> = target_indices
            .values()
            // For each target value
            .map(|target_value_indices| {
                attribute_indices
                    .iter()
                    // For each attribute
                    .map(|attribute_index| {
                        // let mut attribute_value_counts = BTreeMap::new();
                        // For each entry that has given target value
                        let attribute_value_counts = target_value_indices
                            .iter()
                            .map(|index| data.get_value(*attribute_index, *index).assume_nominal())
                            .fold(BTreeMap::new(), |mut counts, attribute_value| {
                                // Count the number each feature occurs for each attribute
                                counts
                                    .entry(attribute_value)
                                    .and_modify(|count| *count += 1)
                                    .or_insert(1u32);
                                counts
                            });

                        // Number of values attribute can take on
                        let n = data.get_attributes()[*attribute_index]
                            .assume_nominal()
                            .size();

                        // We need to calculate probability for all features of attribute
                        // even if the count is 0
                        (0..n)
                            .map(|attribute_value| {
                                *attribute_value_counts
                                    .get(&(attribute_value as u8))
                                    .unwrap_or(&0) as usize
                            })
                            .map(|count| {
                                (count + k) as f32 / (target_value_indices.len() + (n * k)) as f32
                            })
                            .collect()
                    })
                    // Collect into a Vec<Vec<f32>> for given target value
                    .collect::<Vec<_>>()
            })
            // Collect into a Vec<Vec<Vec<f32>>>
            .collect();

        Self {
            probability_target,
            probability_given,
        }
    }

    /// Not yet implemented!
    /// Will handle conversion from numeric to nominal data then train model on that data
    pub fn new_from_numeric(data: DataSet, target: usize) -> Self {
        todo!("{:?} {}", data, target)
    }

    /// Query an entry. Assumes entry matches a data entry of the training set
    /// with target value included. Meant for to be used for testing
    fn query(&self, entry: &[u8], target: usize) -> u8 {
        // For each target value
        (0..self.probability_given.len())
            .map(|target_value| {
                let probability = (0..entry.len())
                    .filter(|&index| index != target)
                    .map(|attribute_index| {
                        self.probability_given[target_value][attribute_index]
                            [entry[attribute_index] as usize]
                    })
                    // Rust struggles with inferring type for product() and sum()
                    .product::<f32>()
                    * self.probability_target[target_value];
                (target_value, probability)
            })
            // In this context _ is used to tell the compiler you don't need that variable
            // Here we just need to compare the probabilities
            .max_by(|(_, prob_a), (_, prob_b)| prob_a.partial_cmp(prob_b).unwrap())
            .unwrap()
            // Return just the target value
            .0 as u8
    }

    /// Test a set of data with known target values to calculate the accuracy
    /// target is the target index of the test_set
    /// target value is what is considered a positive value for calculations of false positives
    pub fn test(&self, test_set: &DataSet, target: usize, _target_value: u8) {
        // Now  works for any finite set of target features
        let target_feature_size = test_set.get_attributes()[target].assume_nominal().size();
        let count = (0..test_set.get_data_len())
            .map(|data_index| {
                // Collecting the entry for each data index
                (0..test_set.get_len())
                    .map(|attribute_index| {
                        test_set
                            .get_value(attribute_index, data_index)
                            .assume_nominal()
                    })
                    .collect::<Vec<_>>()
            })
            // entry -> (predicted, actual)
            .map(|entry| (self.query(&entry, target) as usize, entry[target] as usize))
            // fold is an iterator consumer that produces a single value, in this case a Confusion Matrix
            // sum, product, join (from itertools), for_each, collect, etc. are other consumers
            // Iterators are lazy and won't do anything unless it's being "consumed"
            // https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.fold
            .fold(
                ConfusionMatrix::new(target_feature_size),
                |mut count, (predicted, actual)| {
                    count.add_prediction(predicted, actual);
                    count
                },
            );

        // Display the statistics deduced from the matrix
        count.display(1);
    }
}
