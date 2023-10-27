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
        // For each predicted value
        (0..self.n)
            .map(|predicted| {
                // For each actual value
                (0..self.n)
                    // (predicted, actual) -> count
                    .map(|actual| self.get(predicted, actual))
                    .map(|count| format!("{:4}", count))
                    .join(" ")
            })
            .for_each(|row| println!("{}", row));
        println!("Accuracy: {:.3}", self.accuracy());
    }

    fn accuracy(&self) -> f32 {
        let correct: u32 = (0..self.n).map(|n| self.get(n, n)).sum();
        let total: u32 = (0..self.n)
            // Iterates over every permutation of two iterators (from itertools)
            .cartesian_product(0..self.n)
            .map(|(predicted, actual)| self.get(predicted, actual))
            .sum::<u32>();
        correct as f32 / total as f32
    }
}
