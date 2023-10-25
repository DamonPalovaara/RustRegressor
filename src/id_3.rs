use std::collections::HashMap;

use crate::{
    all_equal,
    data_set::{Data, DataSet, Nominal},
    majority_vote_ordered, normalize, swap_remove,
};

// Just need IDs for the categories. Could considering using u16 or u8s instead
// layout is used to convert test data
pub struct ProcessedData {
    pub layout: Vec<DataType>,
    // OOD structure: [[1234],[1234],...]
    // DOD structure: [[1111],[2222], ...]
    data: Vec<Vec<u32>>,
}

impl ProcessedData {
    fn data_len(&self) -> usize {
        self.data[0].len()
    }

    fn attribute_len(&self) -> usize {
        self.data.len()
    }

    fn get_attribute(&self, index: usize) -> &[u32] {
        &self.data[index]
    }

    // Returning error here so I could hunt down problem area
    fn get_value(&self, attribute: usize, index: usize) -> Result<u32, &str> {
        // Question mark operator is short hand for returning the error
        let attribute_data = self.data.get(attribute).ok_or("Attribute out of bounds")?;
        let value = attribute_data.get(index).ok_or("Index out of bounds")?;
        Ok(*value)
    }
}

impl ProcessedData {
    pub fn import(data_set: &DataSet) -> Self {
        let (layout, data): (Vec<_>, Vec<_>) = data_set
            .get_attributes()
            .iter()
            .map(|attribute| match attribute.get_data() {
                Data::Nominal(data) => Self::process_nominal(data),
                Data::Numeric(data) | Data::Real(data) => Self::process_numeric(data),
                Data::Date(_) => panic!("Teach me how to process date data"),
                Data::String(_) => panic!("Teach me how to process strings"),
                Data::NotImplemented(data_type) => panic!("{} needs to be implemented!", data_type),
            })
            .unzip();
        Self { layout, data }
    }

    // Test data needs to be placed into same buckets as training data so needs it's layout
    pub fn import_test_data(data_set: &DataSet, layout: &[DataType]) -> Self {
        let data = data_set
            .get_attributes()
            .iter()
            .zip(layout.iter())
            .map(|(attribute, layout)| match attribute.get_data() {
                Data::Nominal(data) => Self::process_nominal(data).1,
                Data::Numeric(data) | Data::Real(data) => {
                    Self::process_numeric_from_layout(data, layout)
                }
                _ => panic!("Haven't implemented other types yet!"),
            })
            .collect();

        Self {
            data,
            // Could optimize this out but compiler probably doing so already
            layout: layout.to_vec(),
        }
    }

    fn process_numeric_from_layout(data: &[f32], layout: &DataType) -> Vec<u32> {
        let parser = match layout {
            DataType::Numeric(numeric_type) => numeric_type,
            // unreachable!() panics when reached
            // unreachable_unchecked!() allows for this match statement to be optimized away
            // but requires unsafe rust to do so and causes undefined behavior if reached
            _ => unreachable!(),
        };
        data.iter().map(|x| parser.convert(*x)).collect()
    }

    fn process_nominal(data: &Nominal) -> (DataType, Vec<u32>) {
        let layout = DataType::new_nominal(data.get_map());
        (
            layout,
            data.get_data().iter().map(|value| *value as u32).collect(),
        )
    }

    fn process_numeric(data: &[f32]) -> (DataType, Vec<u32>) {
        let buckets = (data.len() as f32).sqrt() as u32;

        let min = data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            .to_owned();
        let max = data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            .to_owned();

        let data_type = NumericType { min, max, buckets };
        let data = data.iter().map(|value| data_type.convert(*value)).collect();
        let layout = DataType::Numeric(data_type);

        (layout, data)
    }
}

#[derive(Debug, Clone)]
pub enum DataType {
    Numeric(NumericType),
    Nominal(NominalType),
}

impl DataType {
    fn new_nominal(map: &HashMap<String, u8>) -> Self {
        Self::Nominal(NominalType { map: map.clone() })
    }
}

#[derive(Debug, Clone)]
pub struct NumericType {
    min: f32,
    max: f32,
    buckets: u32,
}

impl NumericType {
    fn convert(&self, data: f32) -> u32 {
        let value = (normalize(self.min, self.max, data) * self.buckets as f32) as i64;
        if value >= self.buckets as i64 {
            self.buckets - 1
        } else if value < 0 {
            0
        } else {
            value as u32
        }
    }
}
#[derive(Debug, Clone)]
pub struct NominalType {
    map: HashMap<String, u8>,
}

impl NominalType {
    pub fn convert(&self, value: &str) -> u32 {
        *self.map.get(value).expect("Key not in map") as u32
    }
}

// (split value) -> (target value) -> (count of target value for split value)
struct Counter {
    counts: HashMap<u32, HashMap<u32, u32>>,
    len: usize,
}

impl Counter {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
            len: 0,
        }
    }

    fn insert(&mut self, split_value: u32, target_value: u32) {
        self.len += 1;
        self.counts
            .entry(split_value)
            .and_modify(|target_count| {
                target_count
                    .entry(target_value)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            })
            .or_insert(HashMap::from([(target_value, 1u32)]));
    }

    fn entropy(&self) -> f32 {
        self.counts
            .values()
            .map(|target_count| {
                let partition_size = target_count.values().sum::<u32>();
                let partition_entropy = target_count
                    .values()
                    .map(|count| *count as f32 / partition_size as f32)
                    .map(|probability| -probability * probability.log2())
                    .sum::<f32>();
                (partition_size as f32 / self.len as f32) * partition_entropy
            })
            .sum()
    }
}

pub struct ID3 {
    root: Node,
}

impl ID3 {
    pub fn new() -> Self {
        let root = Node::Internal(InternalNode::new());
        Self { root }
    }

    pub fn train(&mut self, data: &ProcessedData, target: usize) {
        let attribute_len = data.attribute_len();
        let data_len = data.data_len();
        let majority_element =
            majority_vote_ordered((0..data_len).map(|index| data.get_value(target, index)))
                .unwrap();
        match &mut self.root {
            Node::Internal(node) => node.value = majority_element,
            _ => unreachable!(),
        }
        let indices: Vec<_> = (0..data_len).collect();
        let mut attributes: Vec<_> = (0..attribute_len)
            .filter(|index| *index != target)
            .collect();
        self.root.train(&indices, data, &mut attributes, target);
    }

    pub fn test(&self, test_data: &ProcessedData, target: usize) {
        let correct_count: usize = (0..test_data.data_len())
            .map(|data_index| {
                (0..test_data.attribute_len())
                    .map(|attribute_index| {
                        test_data.get_value(attribute_index, data_index).unwrap()
                    })
                    .collect()
            })
            .map(|entry: Vec<u32>| (self.query(&entry), entry[target]))
            .filter(|(predicted, actual)| predicted == actual)
            .map(|_| 1)
            .sum();
        println!(
            "Accuracy {}",
            correct_count as f32 / test_data.data_len() as f32
        );
    }

    pub fn query(&self, data: &[u32]) -> u32 {
        self.root.query(data)
    }

    pub fn display(&self) {
        self.root.display(0);
    }
}

impl Default for ID3 {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
enum Node {
    Internal(InternalNode),
    Leaf(LeafNode),
}

impl Node {
    fn train(
        &mut self,
        indices: &[usize],
        data: &ProcessedData,
        attributes: &mut [usize],
        target: usize,
    ) {
        match self {
            Self::Internal(node) => node.train(indices, data, attributes, target),
            _ => unreachable!("Tried training a leaf-node"),
        }
    }

    fn display(&self, depth: usize) {
        match self {
            Self::Internal(node) => node.display(depth),
            Self::Leaf(node) => node.display(depth),
        }
    }

    fn query(&self, data: &[u32]) -> u32 {
        match self {
            Self::Internal(node) => node.query(data),
            Self::Leaf(node) => node.query(),
        }
    }

    fn has_key(&self, key: u32) -> bool {
        match self {
            Self::Internal(node) => node.key == key,
            Self::Leaf(node) => node.key == key,
        }
    }
}

// Category is the category the children got split on
// Key is the category value of the parents category
#[derive(Debug)]
struct InternalNode {
    category: usize,
    key: u32,
    // For when child with attribute key is missing. Set using majority vote
    value: u32,
    children: Vec<Node>,
}

impl InternalNode {
    fn new() -> Self {
        Self {
            category: 0,
            key: 0,
            value: 0,
            children: Vec::new(),
        }
    }

    fn from_key_value(key: u32, value: u32) -> Self {
        Self {
            category: 0,
            key,
            value,
            children: Vec::new(),
        }
    }

    fn display(&self, depth: usize) {
        (0..depth).for_each(|_| print!(" "));
        self.children
            .iter()
            .for_each(|child| child.display(depth + 1));
        println!("[{} {}] {}", self.category, self.key, depth);
    }

    fn query(&self, data: &[u32]) -> u32 {
        let key = data[self.category];
        let next_node = self.children.iter().find(|child| child.has_key(key));
        match next_node {
            Some(node) => node.query(data),
            None => self.value,
        }
    }

    // indices: The data indices to train on
    // data: The raw processed data
    // attributes: Array of attribute indices to train on
    // target: Index of the target attribute
    fn train(
        &mut self,
        indices: &[usize],
        data: &ProcessedData,
        attributes: &mut [usize],
        target: usize,
    ) {
        // attribute: The index into the data array with least entropy
        // attribute_index: Location of attribute in the attribute-array
        let (&attribute, attribute_index, _entropy) = attributes
            .iter()
            .enumerate()
            .map(|(attribute_index, attribute)| {
                let mut counter = Counter::new();
                data.get_attribute(*attribute)
                    .iter()
                    .zip(data.get_attribute(target).iter())
                    .for_each(|(split_value, target_value)| {
                        counter.insert(*split_value, *target_value)
                    });

                (attribute, attribute_index, counter.entropy())
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).expect("Can't compare NaN's!"))
            .unwrap();

        self.category = attribute;

        // Create the arrays for the children to learn from
        let mut children: HashMap<u32, Vec<usize>> = HashMap::new();
        indices
            .iter()
            .map(|index| (index, data.get_value(attribute, *index).unwrap()))
            .for_each(|(index, attribute_value)| {
                children
                    .entry(attribute_value)
                    .and_modify(|indices| indices.push(*index))
                    .or_insert(vec![*index]);
            });

        // Remove the attribute just consumed by this call
        let attributes = swap_remove(attributes, attribute_index);

        // Transform the children-map into nodes
        self.children = children
            .iter()
            .map(|(&attribute, indices)| {
                // Majority vote if no more attributes to split on
                if attributes.is_empty() {
                    Node::Leaf(LeafNode {
                        key: attribute,
                        value: majority_vote_ordered(
                            indices
                                .iter()
                                .map(|index| data.get_value(target, *index).unwrap()),
                        ),
                    })
                }
                // If all target values are the same then return that target value
                else if all_equal(indices.iter().map(|index| data.get_value(target, *index))) {
                    Node::Leaf(LeafNode {
                        key: attribute,
                        value: data.get_value(target, indices[0]).unwrap(),
                    })
                }
                // More training required
                else {
                    let mut child = InternalNode::from_key_value(
                        attribute,
                        majority_vote_ordered(
                            indices
                                .iter()
                                .map(|index| data.get_value(target, *index).unwrap()),
                        ),
                    );
                    child.train(indices, data, attributes, target);
                    Node::Internal(child)
                }
            })
            .collect();
    }
}

#[derive(Debug)]
struct LeafNode {
    key: u32,
    value: u32,
}

impl LeafNode {
    fn display(&self, _depth: usize) {
        println!("Leaf-node! {} {}", self.key, self.value);
    }

    fn query(&self) -> u32 {
        self.value
    }
}

// Unit testing is super easy!
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_three_partitions_equal() {
        let mut counter = Counter::new();
        counter.insert(1, 1);
        counter.insert(2, 1);
        counter.insert(3, 1);

        let result = counter.entropy();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_entropy_two_partitions_half() {
        let mut counter = Counter::new();
        counter.insert(1, 1);
        counter.insert(1, 2);
        counter.insert(2, 1);
        counter.insert(2, 2);

        let result = counter.entropy();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_entropy_thirds() {
        let mut counter = Counter::new();
        counter.insert(1, 1);
        counter.insert(1, 1);
        counter.insert(1, 2);
        counter.insert(1, 2);
        counter.insert(1, 3);
        counter.insert(1, 3);

        let result = counter.entropy();
        assert!((result - 1.58496250072).abs() <= f32::EPSILON);
    }
}
