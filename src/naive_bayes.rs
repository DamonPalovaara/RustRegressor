use std::collections::BTreeMap;

use crate::{data_set::DataSet, test_statistics::TestStats};

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
        let mut map = BTreeMap::<_, Vec<usize>>::new();
        (0..data.get_data_len())
            .map(|index| (index, data.get_value(target, index).assume_nominal()))
            .for_each(|(index, target)| {
                map.entry(target)
                    .and_modify(|entry| entry.push(index))
                    .or_insert(vec![index]);
            });

        // target value -> probability of target value
        let probability_target: Vec<_> = map
            .values()
            .map(|indices| indices.len() as f32 / data.get_data_len() as f32)
            .collect();

        // attribute indices to iterate over
        let attribute_indices: Vec<_> = (0..data.get_len())
            .filter(|&index| index != target)
            .collect();

        // array[target_value][attribute_index][attribute_value] = P(target_value | attribute_value)
        // This is nasty, I'm sorry. I probably should just be using for loops here
        let probability_given: Vec<_> = map
            .values()
            // For each target value
            .map(|target_value_indices| {
                attribute_indices
                    .iter()
                    // For each attribute
                    .map(|attribute_index| {
                        let mut attribute_value_counts = BTreeMap::new();
                        // For each entry that has given target value
                        target_value_indices
                            .iter()
                            .map(|index| data.get_value(*attribute_index, *index).assume_nominal())
                            .for_each(|attribute_value| {
                                // Count the number each attribute value occurs for each attribute
                                attribute_value_counts
                                    .entry(attribute_value)
                                    .and_modify(|count| *count += 1)
                                    .or_insert(1u32);
                            });

                        // Number of values attribute can take on
                        let n = data.get_attributes()[*attribute_index]
                            .assume_nominal()
                            .size();

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
                (
                    target_value,
                    (0..entry.len())
                        .filter(|&index| index != target)
                        .enumerate()
                        .map(|(attribute_index, _)| {
                            self.probability_given[target_value][attribute_index]
                                .get(entry[attribute_index] as usize)
                                .unwrap_or(&0.0)
                        })
                        .product::<f32>()
                        * self.probability_target[target_value],
                )
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0 as u8
    }

    /// Test a set of data with known target values to calculate the accuracy
    /// target is the target index of the test_set
    /// target value is what is considered a positive value for calculations of false positives
    pub fn test(&self, test_set: &DataSet, target: usize, target_value: u8) {
        // This is assuming target feature is boolean, confusion matrix isn't generalize over n elements
        let mut count = TestStats::default();
        (0..test_set.get_data_len())
            .map(|data_index| {
                (0..test_set.get_len())
                    .map(|attribute_index| {
                        test_set
                            .get_value(attribute_index, data_index)
                            .assume_nominal()
                    })
                    .collect::<Vec<_>>()
            })
            .for_each(|entry| {
                if entry[target] == target_value {
                    // Should be positive
                    match self.query(&entry, target) == entry[target] {
                        true => count.add_true_positive(),
                        false => count.add_false_negative(),
                    }
                } else {
                    // Should be negative
                    match self.query(&entry, target) == entry[target] {
                        true => count.add_true_negative(),
                        false => count.add_false_positive(),
                    }
                }
            });
        count.print_stats()
    }
}
