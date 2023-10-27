use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use utc_dt::time::UTCTimestamp;

// derive(A) is a macro to derive trait A
// Derivable traits normally require that every field has trait A
#[derive(Debug)]
pub struct DataSet {
    relation: String,
    // This is the classic example of DoD. I'm keeping data of the same type packed together
    // 1234123412341234 vs 1111222233334444
    attributes: Vec<Attribute>,
}

impl DataSet {
    // Self is shorthand for the type you're implementing, in this case Self == DataSet
    // Self is also conscious of generics e.g. Self == Foo<T> if you're implementing
    // Foo over a generic type T
    pub fn import(file_path: &str) -> Result<Self, String> {
        let raw_data = fs::read_to_string(file_path).map_err(|_| "File not found".to_string())?; //.expect("Unable to read file");

        let relation_index = raw_data.find("@relation").expect("No @relation tag found");
        let relation = Parser::new(&raw_data, relation_index + 9)
            .parse_string()
            .to_owned();

        let mut data_index = raw_data.find("@data").expect("No @data tag found");

        let mut attributes: Vec<_> = raw_data[0..data_index]
            .lines()
            // Remove prefix whitespace. If only whitespace then 0 out size so it gets filtered out
            .map(|line| {
                let start_index = line
                    .find(|c: char| !c.is_whitespace())
                    .unwrap_or(line.len());
                &line[start_index..]
            })
            .filter(|line| !line.is_empty() && !line.starts_with('%'))
            .map(|line| Parser::new(line, 0))
            .map(|mut parser| (parser.parse_string(), parser))
            .filter_map(|(first_word, mut parser)| match first_word {
                "@attribute" => Some(parser.parse_attribute()),
                _ => None,
            })
            .collect();

        // Finding where actual data starts
        data_index = (data_index + 5)
            + raw_data[(data_index + 5)..]
                .find(|c: char| !c.is_whitespace())
                .unwrap_or(raw_data.len() - (data_index + 5));

        raw_data[data_index..].lines().for_each(|line| {
            line.split(',')
                .enumerate()
                .for_each(|(index, value)| attributes[index].parse_value(value))
        });
        Ok(Self {
            attributes,
            relation,
        })
    }

    pub fn display(&self) {
        println!("Relation: {}", self.relation);

        self.attributes
            .iter()
            .for_each(|attribute| match &attribute.data {
                Data::Nominal(data) => println!("{}: {:?}", attribute.label, &data.fields),
                Data::Numeric(data) | Data::Real(data) => println!(
                    "{}: [min: {}, max: {}]",
                    attribute.label,
                    data.iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(&0.0),
                    data.iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(&0.0)
                ),
                _ => (),
            })
    }

    pub fn get_value(&self, attribute_index: usize, data_index: usize) -> DataEntry {
        self.attributes[attribute_index].get_value(data_index)
    }

    /// Returns the number of columns in the data set (the number of descriptive features)
    pub fn get_len(&self) -> usize {
        self.attributes.len()
    }

    pub fn get_data_len(&self) -> usize {
        if self.get_attributes().is_empty() {
            0
        } else {
            match self.get_attributes()[0].get_data() {
                Data::Nominal(nominal) => nominal.get_data().len(),
                Data::Real(data) | Data::Numeric(data) => data.len(),
                _ => panic!("Need to implement other types"),
            }
        }
    }

    pub fn get_attributes(&self) -> &[Attribute] {
        &self.attributes
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum DataEntry {
    Numeric(f32),
    Nominal(u8),
}

impl DataEntry {
    pub fn assume_numeric(&self) -> f32 {
        match self {
            Self::Numeric(value) => *value,
            _ => panic!("Assumed numeric but is nominal!"),
        }
    }

    pub fn assume_nominal(&self) -> u8 {
        match self {
            Self::Nominal(value) => *value,
            _ => panic!("Assumed nominal but is numeric!"),
        }
    }
}

// Can't derive because f32 isn't EQ. This is just a marker trait so no method to implement
impl Eq for DataEntry {}

// Can't derive because f32 isn't Hash, need to convert to bytes first then hash
impl Hash for DataEntry {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Self::Numeric(data) => data.to_ne_bytes().hash(state),
            Self::Nominal(data) => data.hash(state),
        }
    }
}

// Rust enums are algebraic data types en.wikipedia.org/wiki/Algebraic_data_type
// They can take on any of the variants
// It takes on the size of the largest variant plus a byte to store which variant
// Sometimes the byte gets optimized away e.g. Option<NonNullPointer> allows None to be 0
#[derive(Debug)]
pub enum Data {
    // This is the data that's tightly packed together
    // This makes the CPU cache very happy :)
    Numeric(Vec<f32>),
    Real(Vec<f32>),
    // Wrap Nominal because it's a big type (72 bytes!)
    Nominal(Box<Nominal>),
    // This makes the CPU cache sad :( will change to IDs in future
    String(Vec<String>),
    Date(Vec<Date>),
    NotImplemented(String),
}

#[derive(Debug)]
pub struct Nominal {
    // fields[id] -> field value
    fields: Vec<String>,
    // vector of ids (makes computer cache much happier over vector of strings)
    data: Vec<u8>,
    // map[field value] -> id
    map: HashMap<String, u8>,
}

impl Nominal {
    fn new(fields: Vec<String>) -> Self {
        let data = Vec::new();
        let map = HashMap::from_iter(
            fields
                .iter()
                .enumerate()
                .map(|(index, field)| (field.into(), index as u8)),
        );

        Self { fields, data, map }
    }

    fn push(&mut self, value: &str) {
        // Could add new field here rather than panicking
        let id = self.map.get(value).expect("Value does not match any field");

        self.data.push(*id);
    }

    pub fn size(&self) -> usize {
        self.map.len()
    }

    pub fn get(&self, index: usize) -> &str {
        &self.fields[self.data[index] as usize]
    }

    pub fn get_fields(&self) -> &[String] {
        &self.fields
    }

    pub fn get_data(&self) -> &[u8] {
        &self.data
    }

    pub fn get_map(&self) -> &HashMap<String, u8> {
        &self.map
    }
}

#[derive(Debug)]
pub struct Attribute {
    pub label: String,
    pub data: Data,
}

impl Attribute {
    fn parse_value(&mut self, raw_data: &str) {
        let mut parser = Parser::new(raw_data, 0);
        match &mut self.data {
            Data::Numeric(data) => data.push(raw_data.parse().unwrap()),
            Data::Real(data) => data.push(raw_data.parse().unwrap()),
            Data::Nominal(data) => data.push(parser.parse_string()),
            Data::String(data) => data.push(parser.parse_string().to_owned()),
            Data::Date(data) => data.push(parser.parse_date()),
            Data::NotImplemented(data_type) => panic!("{} hasn't been implemented", data_type),
        }
    }

    pub fn assume_nominal(&self) -> &Nominal {
        match &self.data {
            Data::Nominal(nominal) => nominal,
            _ => panic!("Made wrong assumption"),
        }
    }

    pub fn get_data(&self) -> &Data {
        &self.data
    }

    pub fn get_value(&self, index: usize) -> DataEntry {
        match self.get_data() {
            Data::Numeric(data) | Data::Real(data) => DataEntry::Numeric(data[index]),
            Data::Nominal(nominal) => DataEntry::Nominal(nominal.data[index]),
            _ => panic!("Need to implement type!"),
        }
    }
}

// QoL wrapper around UTCTimestamp
// Not sure if nano-second precision is necessary so will consider storing as 1 u64
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Date(UTCTimestamp);

// Generic over a lifetime 'a
// Parser is given the same lifetime as the source string
// e.g. if the source string is 'static then 'a == 'static
// The burrow checker will guarantee that the source string out lives the Parser
// That means no worry of a dangling pointer guaranteed by the compiler
struct Parser<'a> {
    raw_data: &'a str,
    index: usize,
}

impl<'a> Parser<'a> {
    // Note that Self == Parser<'a>
    fn new(raw_data: &'a str, index: usize) -> Self {
        Self { raw_data, index }
    }

    fn parse_date(&mut self) -> Date {
        // todo!() just panics if reached, this allows the code to compile
        todo!("Time to implement parse_date()")
    }

    fn parse_string(&mut self) -> &'a str {
        let mut start_index = self.index
            + self.raw_data[self.index..]
                .find(|c: char| !c.is_whitespace())
                .expect("End of file reached!");

        let start_char = self
            .raw_data
            .get(start_index..=start_index)
            .expect("Issue reading character");
        let length = match start_char {
            "'" => {
                start_index += 1;
                1 + self.raw_data[(start_index + 1)..]
                    .find('\'')
                    .expect("Missing closing single quote")
            }
            "\"" => {
                start_index += 1;
                1 + self.raw_data[(start_index + 1)..]
                    .find('"')
                    .expect("Missing closing double quote")
            }
            _ => self.raw_data[(start_index)..]
                .find(|c: char| c.is_whitespace())
                .unwrap_or(self.raw_data.len() - start_index),
        };

        let next_white_space = self.raw_data[(start_index + length)..]
            .find(|c: char| c.is_whitespace())
            .unwrap_or(self.raw_data.len());

        self.index = start_index + length + next_white_space;

        &self.raw_data[start_index..(start_index + length)]
    }

    fn parse_attribute(&mut self) -> Attribute {
        let label = self.parse_string().to_owned();
        let start_index = self.raw_data[self.index..]
            .find(|c: char| !c.is_whitespace())
            .expect("End of file reached!");
        let start_char = self.raw_data[self.index..]
            .get(start_index..=start_index)
            .expect("Error getting first character");
        let data = if start_char == "{" {
            self.parse_nominal()
        } else {
            let data_type = self.parse_string();
            match data_type {
                "numeric" => Data::Numeric(Vec::new()),
                "string" => Data::String(Vec::new()),
                "date" => Data::Date(Vec::new()),
                "real" => Data::Real(Vec::new()),
                _ => Data::NotImplemented(data_type.to_owned()),
            }
        };
        Attribute { label, data }
    }

    fn parse_nominal(&mut self) -> Data {
        self.index += self.raw_data[self.index..]
            .find('{')
            .expect("Could not find opening {");
        let length = self.raw_data[self.index..]
            .find('}')
            .expect("Could not find closing }")
            + 1;
        let fields = self.raw_data[(self.index + 1)..(self.index + length - 1)]
            .split(',')
            .map(|word| Parser::new(word, 0).parse_string().to_owned())
            .collect();
        self.index += length;

        Data::Nominal(Box::new(Nominal::new(fields)))
    }
}
