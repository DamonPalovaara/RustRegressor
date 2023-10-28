use itertools::Itertools;

/// Simplified statistics structure that simply keeps track of right and wrong predictions
#[derive(Default)]
pub struct RightVsWrong {
    right: u64,
    wrong: u64,
}

impl RightVsWrong {
    /// Increase the count of right predictions
    pub fn was_right(&mut self) {
        self.right += 1;
    }

    /// Increase the count of wrong predictions
    pub fn was_wrong(&mut self) {
        self.wrong += 1;
    }

    /// Display the accuracy
    pub fn display(&self) {
        println!(
            "Accuracy {}",
            self.right as f32 / (self.right + self.wrong) as f32
        )
    }
}

/// Generalized Confusion Matrix over n elements
/// Matrix is stored as 1D array of length n * n
/// Indexing is handled by get method
pub struct ConfusionMatrix {
    // No need for a vector here, simple pointer to heap is fine
    // Box is an owned pointer to heap btw (owned meaning it gets dropped once it falls out of scope)
    matrix: Box<[u32]>,
    n: usize,
}

impl ConfusionMatrix {
    pub fn new(n: usize) -> Self {
        let matrix = Box::from_iter(std::iter::repeat(0).take(n * n));
        ConfusionMatrix { matrix, n }
    }

    pub fn add_prediction(&mut self, predicted: usize, actual: usize) {
        debug_assert!(actual < self.n && predicted < self.n);
        let index = (predicted * self.n) + actual;
        self.matrix[index] += 1;
    }

    pub fn get(&self, predicted: usize, actual: usize) -> u32 {
        debug_assert!(actual < self.n && predicted < self.n);
        let index = (predicted * self.n) + actual;
        self.matrix[index]
    }

    pub fn display(&self, _positive_feature: usize) {
        let matrix_string = self
            .matrix
            .chunks(self.n)
            .map(|row| row.iter().map(|count| format!("{:4}", count)).join(" "))
            .join("\n");

        println!("{}", matrix_string);
        println!("Accuracy: {:.3}", self.accuracy());
        // This is only meaningful if the target feature is ordinal
        println!("Distance cost: {}", self.distance_cost());
        println!("Distance squared cost: {}", self.distance_squared_cost());
    }

    pub fn accuracy(&self) -> f32 {
        let correct: u32 = (0..self.n).map(|n| self.get(n, n)).sum();
        let total: u32 = self.matrix.iter().sum::<u32>();
        correct as f32 / total as f32
    }

    // Creates a cost matrix with matrix[n][n] = 0 and matrix[m][n] = |m - n|
    // Smaller score is better
    fn distance_cost(&self) -> i64 {
        (0..self.n)
            .cartesian_product(0..self.n)
            .map(|(predicted, actual)| {
                (
                    predicted as i64,
                    actual as i64,
                    self.get(predicted, actual) as i64,
                )
            })
            .map(|(predicted, actual, count)| count * (predicted - actual).abs())
            .sum()
    }

    // Creates a cost matrix with matrix[n][n] = 0 and matrix[m][n] = (m - n)^2
    // Smaller score is better
    fn distance_squared_cost(&self) -> i64 {
        (0..self.n)
            .cartesian_product(0..self.n)
            .map(|(predicted, actual)| {
                (
                    predicted as i64,
                    actual as i64,
                    self.get(predicted, actual) as i64,
                )
            })
            .map(|(predicted, actual, count)| count * (predicted - actual).pow(2))
            .sum()
    }
}
