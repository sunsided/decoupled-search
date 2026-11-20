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
    /// The price of each rank in [`Prices::ranks`], never [`UNREACHED`];
    /// empty if every reached state has price 0, which saves the prices in
    /// reachability-only search and for leaves without costs.
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

        if out.prices.iter().all(|&p| p == 0) {
            out.prices = Vec::new();
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
        let zeros = core::iter::repeat_n(0, self.ranks.len() - self.prices.len());
        self.ranks
            .iter()
            .map(|&r| usize::from(r))
            .zip(self.prices.iter().copied().chain(zeros))
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

        let zeros = [0, UNREACHED, 0];
        let sparse = Prices::from_dense(&zeros);
        assert!(sparse.prices.is_empty(), "all-zero prices are not stored");
        assert_eq!(sparse.iter().collect::<Vec<_>>(), [(0, 0), (2, 0)]);
        assert_eq!(sparse.to_dense(zeros.len()), zeros);
    }
}
