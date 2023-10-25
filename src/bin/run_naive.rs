use arff_reader::{data_set::DataSet, naive_bayes::NaiveBayes};
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let train_set = DataSet::import("./datasets/lakes/lakesDiscreteFold1.arff");
    let test_set = DataSet::import("./datasets/lakes/lakesDiscreteFold2.arff");
    for k in 0..=10 {
        let naive_bayes = NaiveBayes::new(&train_set, 18, k);
        println!("K: {}", k);
        naive_bayes.test(&test_set, 18, 1);
        println!();
    }

    println!("Finished in {:?}", now.elapsed())
}
