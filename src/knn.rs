use crate::{
    data_set::{Data, DataEntry, DataSet},
    majority_vote,
    test_statistics::RightVsWrong,
};

/// K-Nearest Neighbors implementation. Works by taking a majority vote of the k-nearest neighbor's target value
pub struct KNN {
    data: DataSet,
}

impl KNN {
    /// Returns KNN instance that will use the data-points in data for queries
    pub fn new(data: DataSet) -> Self {
        Self { data }
    }

    // Returns a vector of indices into the dataset ordered by closest to the test point
    fn nearest_neighbors(&self, test_set: &DataSet, index: usize, target: usize) -> Vec<usize> {
        let attribute_indices: Vec<usize> = (0..self.data.get_attributes().len())
            .filter(|index| *index != target)
            .collect();
        let distances: Vec<Vec<f32>> = attribute_indices
            .iter()
            .map(
                |attribute_index| match self.data.get_attributes()[*attribute_index].get_data() {
                    Data::Numeric(data) => data
                        .iter()
                        .map(|lhs| {
                            let rhs = match test_set.get_value(*attribute_index, index) {
                                DataEntry::Numeric(data) => data,
                                _ => unreachable!(),
                            };
                            distance_squared(*lhs, rhs)
                        })
                        .collect(),
                    Data::Nominal(nominal) => nominal
                        .get_data()
                        .iter()
                        .map(|lhs| {
                            let rhs = match test_set.get_value(*attribute_index, index) {
                                DataEntry::Nominal(data) => data,
                                _ => unreachable!(),
                            };
                            distance_nominal(*lhs, rhs)
                        })
                        .collect(),
                    _ => panic!("Need to implement more types!"),
                },
            )
            .collect();

        let distances: Vec<f32> = (0..self.data.get_data_len())
            .map(|data_index| {
                // Note that there is no need to square-root since it doesn't change ordering
                (0..distances.len())
                    .map(|attribute_index| distances[attribute_index][data_index])
                    .sum::<f32>()
            })
            .collect();

        let mut nearest_neighbors: Vec<usize> = (0..self.data.get_data_len()).collect();
        nearest_neighbors.sort_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap());
        nearest_neighbors
    }

    // Returns predicted target value based on the k nearest neighbors
    fn query_k(&self, nearest_neighbors: &[usize], k: usize, target: usize) -> DataEntry {
        majority_vote(
            nearest_neighbors
                .iter()
                .take(k)
                .map(|index| self.data.get_value(target, *index)),
        )
    }

    /// Runs an accuracy test for each value of k and displays it.
    /// We actually only need to calculate the distances once then query the
    /// k nearest neighbors for each k using the same sorted nearest-neighbors vector
    pub fn test(&self, test_set: &DataSet, target: usize) {
        // Vector of nearest neighbors for each entry of the test data
        let nearest_neighbors: Vec<_> = (0..test_set.get_data_len())
            .map(|index| self.nearest_neighbors(test_set, index, target))
            .collect();

        // Iterate over each possible value of k
        (1..=self.data.get_data_len())
            .map(|k| {
                let mut count = RightVsWrong::default();
                nearest_neighbors
                    .iter()
                    .enumerate()
                    // Worth noting that nearest_neighbors in this context is for the single entry, not all the entries
                    .map(|(index, nearest_neighbors)| {
                        (index, self.query_k(nearest_neighbors, k, target))
                    })
                    .map(|(index, prediction)| (prediction, test_set.get_value(target, index)))
                    .for_each(|(prediction, actual)| match prediction == actual {
                        true => count.was_right(),
                        false => count.was_wrong(),
                    });
                (k, count)
            })
            .for_each(|(k, result)| {
                println!("k: {}", k);
                result.display();
            });
    }
}

/// Three forward slashes creates a doc-comment
/// Doc-comments only appear for public parts so this won't appear in the docs
// Returns distance of 1 if they're different or 0 if they're the same
fn distance_nominal(lhs: u8, rhs: u8) -> f32 {
    match lhs == rhs {
        true => 0.0,
        false => 1.0,
    }
}

// Squared distance of 2 numeric values
fn distance_squared(lhs: f32, rhs: f32) -> f32 {
    (lhs - rhs).powi(2)
}
