use arff_reader::{data_set::DataSet, naive_bayes::NaiveBayes};
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let target_index = 10;
    let train_set = DataSet::import("./test_data/lakesDiscreteFold1.arff").expect("File not found");
    let test_set = DataSet::import("./test_data/lakesDiscreteFold2.arff").expect("File not found");
    (0..=10).for_each(|k| {
        let naive_bayes = NaiveBayes::new(&train_set, target_index, k);
        println!("K: {}", k);
        naive_bayes.test(&test_set, target_index, 1);
        println!();
    });

    println!("Finished in {:?}", now.elapsed())
}
