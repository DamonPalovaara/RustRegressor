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

pub struct ConfusionMatrix {
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
        self.matrix[(actual * self.n) + predicted] += 1;
    }

    fn get(&self, predicted: usize, actual: usize) -> u32 {
        let index = (actual * self.n) + predicted;
        self.matrix[index]
    }

    pub fn display(&self, _positive_feature: usize) {
        (0..self.n).for_each(|row| {
            (0..self.n)
                .map(|column| self.get(row, column))
                .for_each(|count| print!("{:4} ", count));
            println!();
        });
        println!("Accuracy: {:.3}", self.accuracy());
    }

    fn accuracy(&self) -> f32 {
        let correct: u32 = (0..self.n).map(|n| self.get(n, n)).sum();
        let total: u32 = (0..self.n)
            .map(|row| (0..self.n).map(|column| self.get(row, column)).sum::<u32>())
            .sum::<u32>();
        correct as f32 / total as f32
    }
}
