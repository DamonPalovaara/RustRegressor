use arff_reader::{data_set::DataSet, linear::LinearModel};

use std::time::Instant;

fn main() {
    let now = Instant::now();

    let target_index = 3;
    let train_set = DataSet::import("./test_data/linear_test.arff").expect("File not found");
    let mut linear_model = LinearModel::new(&train_set, target_index);
    linear_model.regress();

    println!("Finished in {:?}", now.elapsed())
}
