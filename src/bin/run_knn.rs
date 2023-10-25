use arff_reader::{data_set::DataSet, knn::KNN};
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let train_set = DataSet::import("./datasets/lakes/lakesA1.arff");
    let test_set = DataSet::import("./datasets/lakes/lakesA2.arff");
    let knn = KNN::new(train_set);
    knn.test(&test_set, 18);

    println!("Finished in {:?}", now.elapsed())
}