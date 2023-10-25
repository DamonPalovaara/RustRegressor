#[derive(Default, Debug)]
/// TestStats keeps track of true positives, false positives, true negatives, and false negatives
pub struct TestStats {
    true_positive: u64,
    false_positive: u64,
    true_negative: u64,
    false_negative: u64,
}

impl TestStats {
    /// Increase the count of true positives
    pub fn add_true_positive(&mut self) {
        self.true_positive += 1;
    }

    /// Increase the count of false positives
    pub fn add_false_positive(&mut self) {
        self.false_positive += 1;
    }

    /// Increase the count of true negatives
    pub fn add_true_negative(&mut self) {
        self.true_negative += 1;
    }

    /// Increase the count of false negatives
    pub fn add_false_negative(&mut self) {
        self.false_negative += 1;
    }

    /// Display the matrix
    pub fn display_matrix(&self) {
        println!("{} {}", self.true_positive, self.false_positive);
        println!("{} {}", self.false_negative, self.true_negative);
    }

    /// Print stats relevant to matrix
    pub fn print_stats(&self) {
        let size =
            self.true_negative + self.true_positive + self.false_negative + self.false_positive;

        let accuracy = (self.true_positive + self.true_negative) as f32 / size as f32;

        let precision =
            self.true_positive as f32 / (self.true_positive + self.false_positive) as f32;

        let recall = self.true_positive as f32 / (self.true_positive + self.false_negative) as f32;

        println!("Accuracy: {:.3}", accuracy);
        println!("Precision: {:.3}", precision);
        println!("Recall: {:.3}", recall);
    }
}

#[derive(Default)]
/// Simplified statistics structure that simply keeps track of right and wrong predictions
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
            self.right as f64 / (self.right + self.wrong) as f64
        )
    }
}
