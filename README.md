# Getting Started
- Have md reader installed so you can view this file correctly (VS-Code has a plugin!)
- Install Rust by visiting https://www.rust-lang.org/tools/install
- Familiarize yourself with cargo (Rust's package manager) by executing ```cargo --help```

# Basic Commands
To do a basic debug run execute: ```cargo run```

To do a release build add --release flag

If there are multiple binaries you need to add: --bin binary_name

Where binary_name is the binary to execute. Look in src/bin for all binary names (omit the .rs)

To build and view the docs execute: ```cargo doc --open```

Check out the doc comments I wrote for the knn and test_statistics modules!

To run the unit tests execute: ```cargo test```

To check the pickiest of lints execute: ```cargo clippy```

To remove all compiled components execute: ```cargo clean```

Useful for minimizing the size before turning in

# Notes on Each Binary

If you have multiple binary files in a Rust project you must use a --bin flag to select which binary to run. Below are notes on executing each binary.

## ID3
To run execute:
```bash
cargo run --release --bin run_id3
```

## KNN
To run knn execute:
```bash
cargo run --release --bin run_knn
```
KNN will display the accuracy for each k so the output will be very long. To remedy this you can export to a file instead by running the following command:
```bash
cargo run --release --bin run_knn > export
```
Where export is the name of the file you're exporting to.

## Naive Bayes 
To run execute:
```bash
cargo run --release --bin run_naive 
```
It will run multiple models with different smoothing values set and display each models performance