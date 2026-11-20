//! Compact storage of one leaf's prices in a stored decoupled state.

use alloc::vec;
use alloc::vec::Vec;

use super::leaf::{Price, UNREACHED};

/// The reached leaf states of one leaf with their prices, sorted by rank.
///
/// Stored decoupled states keep only reached leaf states: two bytes of rank
/// and eight bytes of price each, instead of eight bytes for every leaf
/// state. Ranks fit in `u16` because a leaf has at most
/// [`crate::MAX_LEAF_STATES`] (`2^16`) states. Leaf reachability works on the
/// dense form ([`Prices::to_dense`]), which a successor builds only while it
/// is being computed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Prices {
    /// Ranks of the reached leaf states, strictly ascending.
    ranks: Vec<u16>,
    /// The price of each rank in [`Prices::ranks`], never [`UNREACHED`].
    prices: Vec<Price>,
}

impl Prices {
    /// Keeps the reached entries of a dense price vector.
    ///
    /// # Panics
    ///
    /// Panics if a reached rank does not fit in `u16`, which the leaf size
    /// limit rules out.
    pub(crate) fn from_dense(dense: &[Price]) -> Self {
        let mut out = Self::default();
        for (rank, &price) in dense.iter().enumerate() {
            if price != UNREACHED {
                out.ranks
                    .push(u16::try_from(rank).expect("leaf ranks fit in u16"));
                out.prices.push(price);
            }
        }

        out
    }

    /// The dense price vector over `size` leaf states, [`UNREACHED`] for
    /// states not stored.
    pub(crate) fn to_dense(&self, size: usize) -> Vec<Price> {
        let mut dense = vec![UNREACHED; size];
        for (rank, price) in self.iter() {
            dense[rank] = price;
        }

        dense
    }

    /// The reached leaf states as `(rank, price)`, by ascending rank.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, Price)> + '_ {
        self.ranks
            .iter()
            .map(|&r| usize::from(r))
            .zip(self.prices.iter().copied())
    }

    /// Whether every leaf state reached in `other` is reached in `self` at no
    /// higher price. Unreached states compare as infinite, so states reached
    /// only in `self` do not matter.
    pub(crate) fn at_most(&self, other: &Self) -> bool {
        let mut mine = self.iter().peekable();
        other.iter().all(|(rank, price)| {
            while mine.next_if(|&(r, _)| r < rank).is_some() {}
            mine.next_if(|&(r, _)| r == rank)
                .is_some_and(|(_, p)| p <= price)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense round trips keep exactly the reached entries.
    #[test]
    fn dense_round_trip() {
        let dense = [UNREACHED, 3, UNREACHED, 0, 7];
        let sparse = Prices::from_dense(&dense);
        assert_eq!(sparse.iter().collect::<Vec<_>>(), [(1, 3), (3, 0), (4, 7)]);
        assert_eq!(sparse.to_dense(dense.len()), dense);
    }

    /// `at_most` matches the dense pointwise comparison.
    #[test]
    fn at_most_matches_dense_comparison() {
        let vectors = [
            [UNREACHED, UNREACHED, UNREACHED],
            [0, UNREACHED, 2],
            [0, 1, 2],
            [1, 1, UNREACHED],
            [UNREACHED, 0, 3],
        ];
        for a in &vectors {
            for b in &vectors {
                let dense = a.iter().zip(b).all(|(x, y)| x <= y);
                let sparse = Prices::from_dense(a).at_most(&Prices::from_dense(b));
                assert_eq!(sparse, dense, "{a:?} <= {b:?}");
            }
        }
    }
}
