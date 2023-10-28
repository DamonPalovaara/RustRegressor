use arff_reader::{data_set::DataSet, knn::KNN};
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let target = 18;
    let train_set = DataSet::import("./test_data/lakesA1.arff").expect("File not found");
    let test_set = DataSet::import("./test_data/lakesA2.arff").expect("File not found");
    let knn = KNN::new(train_set);
    knn.test(&test_set, target);

    println!("Finished in {:?}", now.elapsed())
}
