use std::time::Instant;

use arff_reader::{
    data_set::DataSet,
    id_3::{ProcessedData, ID3},
};

fn main() {
    let now = Instant::now();

    let train_set = DataSet::import("./test_data/lakesDiscreteFold1.arff").expect("File not found");
    let test_set = DataSet::import("./test_data/lakesDiscreteFold2.arff").expect("File not found");
    let target = 10;
    let mut id3 = ID3::default();
    let training_data = ProcessedData::import(&train_set);
    let testing_data = ProcessedData::import_test_data(&test_set, training_data.layout.as_slice());
    id3.train(&training_data, target);
    id3.test(&testing_data, target);

    println!("Finished in {:?}", now.elapsed())
}
