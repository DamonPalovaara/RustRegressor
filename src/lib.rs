use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

pub mod data_set;
pub mod id_3;
pub mod knn;
pub mod naive_bayes;
pub mod test_statistics;

/// Swap out the value with the last then return slice without last element
#[inline]
pub fn swap_remove<T>(data: &mut [T], index: usize) -> &mut [T] {
    data.swap(index, data.len() - 1);
    data.split_last_mut().unwrap().1
}

/// Normalize value with 0.0 representing min and 1.0 representing max
#[inline]
pub fn normalize(min: f32, max: f32, value: f32) -> f32 {
    (value - min) / (max - min)
}

/// Returns true if all elements in iter are equal
#[inline]
pub fn all_equal<I, T>(iter: I) -> bool
where
    I: IntoIterator<Item = T>,
    T: PartialEq,
{
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    iter.all(|elem| elem == first)
}

/// Returns highest ordered element with highest votes
#[inline]
pub fn majority_vote_ordered<I, T>(iter: I) -> T
where
    I: IntoIterator<Item = T>,
    // Eq & Order for BTreeMap, Copy to return owned T
    T: Eq + Ord + Copy,
{
    // BTree here to resolve inconsistency issue
    let mut counts = BTreeMap::new();

    iter.into_iter().for_each(|element| {
        counts
            .entry(element)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    });

    // Last element with max value is returned here.
    // Could reverse iterator to return lowest ordered element with max value
    counts.iter().max_by_key(|a| a.1).unwrap().0.to_owned()
}

/// Returns the element with most votes, tie breaks are random
#[inline]
pub fn majority_vote<I, T>(iter: I) -> T
where
    I: IntoIterator<Item = T>,
    T: Eq + Hash + Copy,
{
    // Hash maps are randomly seeded
    let mut counts = HashMap::new();

    iter.into_iter().for_each(|element| {
        counts
            .entry(element)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    });

    counts.iter().max_by_key(|a| a.1).unwrap().0.to_owned()
}
