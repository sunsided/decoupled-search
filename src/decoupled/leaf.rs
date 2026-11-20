//! One compiled leaf factor: ranking of leaf states and leaf reachability.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Fact, OpId, Value, VarId};

use super::Prices;

/// The cost of reaching a leaf state; [`UNREACHED`] when it is not reached.
pub(crate) type Price = u64;

/// The price of a leaf state that is not reached.
pub(crate) const UNREACHED: Price = Price::MAX;

/// How a leaf state was first reached at its final price during a closure:
/// from leaf state `from` (a rank) by leaf action `op`.
pub(crate) type Step = Option<(usize, OpId)>;

/// A condition or assignment on one leaf variable.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LeafFact {
    /// Position of the variable in its leaf, as listed in [`crate::Factoring::leaves`].
    pub(crate) slot: usize,
    /// The value, already widened for rank arithmetic.
    pub(crate) value: usize,
}

/// The part of an operator that reads or writes one leaf.
#[derive(Clone, Debug)]
pub(crate) struct LeafPart {
    /// Preconditions on this leaf's variables.
    pub(crate) pre: Vec<LeafFact>,
    /// Effects on this leaf's variables, at most one per variable.
    pub(crate) eff: Vec<LeafFact>,
}

/// An operator whose effects all lie in one leaf and whose preconditions lie
/// in the center and that leaf.
#[derive(Clone, Debug)]
pub(crate) struct LeafAction {
    /// The operator in the original task.
    pub(crate) op: OpId,
    /// The operator cost.
    pub(crate) cost: Price,
    /// Preconditions on the center; `var` is the center slot, not the task variable.
    pub(crate) center_pre: Vec<Fact>,
    /// Preconditions and effects on the leaf.
    pub(crate) part: LeafPart,
}

/// A leaf factor whose states are mixed-radix ranks over its variables.
#[derive(Clone, Debug)]
pub(crate) struct Leaf {
    /// The task variable of each slot.
    pub(crate) vars: Vec<VarId>,
    /// Rank weight of each slot: the product of the domain sizes of all earlier slots.
    pub(crate) strides: Vec<usize>,
    /// Domain size of each slot.
    pub(crate) domains: Vec<usize>,
    /// Number of leaf states, the product of [`Leaf::domains`]. At most
    /// [`crate::MAX_LEAF_STATES`].
    pub(crate) size: usize,
    /// Rank of the leaf's part of the initial state.
    pub(crate) initial: usize,
    /// The task goal restricted to this leaf.
    pub(crate) goal: Vec<LeafFact>,
    /// All leaf actions of this leaf.
    pub(crate) actions: Vec<LeafAction>,
}

impl Leaf {
    /// The value of slot `slot` in leaf state `rank`.
    fn value(&self, rank: usize, slot: usize) -> usize {
        rank / self.strides[slot] % self.domains[slot]
    }

    /// Whether leaf state `rank` satisfies all `facts`.
    pub(crate) fn satisfies(&self, rank: usize, facts: &[LeafFact]) -> bool {
        facts.iter().all(|f| self.value(rank, f.slot) == f.value)
    }

    /// The leaf state after applying the effects `eff` to `rank`.
    pub(crate) fn apply(&self, rank: usize, eff: &[LeafFact]) -> usize {
        eff.iter().fold(rank, |r, f| {
            let old = self.value(r, f.slot);
            r - old * self.strides[f.slot] + f.value * self.strides[f.slot]
        })
    }

    /// Price vector with only the initial leaf state reached, at price 0.
    pub(crate) fn initial_prices(&self) -> Vec<Price> {
        let mut prices = vec![UNREACHED; self.size];
        prices[self.initial] = 0;
        prices
    }

    /// The prices after a global action with leaf part `part`: reached states
    /// that satisfy its preconditions, mapped through its effects, keeping the
    /// cheapest price when two states map to the same one.
    pub(crate) fn transfer(&self, part: &LeafPart, prices: &[Price]) -> Vec<Price> {
        let mut next = vec![UNREACHED; self.size];
        for (rank, &price) in prices.iter().enumerate() {
            if price == UNREACHED || !self.satisfies(rank, &part.pre) {
                continue;
            }

            let target = self.apply(rank, &part.eff);
            next[target] = next[target].min(price);
        }

        next
    }

    /// A reached leaf state of `prices` that [`Leaf::transfer`] maps onto
    /// `seed` at exactly `price`.
    ///
    /// # Panics
    ///
    /// Panics if there is none, which means `price` did not come from a
    /// transfer of `prices`.
    pub(crate) fn transfer_source(
        &self,
        part: &LeafPart,
        prices: &[Price],
        seed: usize,
        price: Price,
    ) -> usize {
        (0..self.size)
            .find(|&rank| {
                prices[rank] == price
                    && self.satisfies(rank, &part.pre)
                    && self.apply(rank, &part.eff) == seed
            })
            .expect("a transferred price has a source leaf state")
    }

    /// Leaf reachability under a fixed center state: Dijkstra from all
    /// reached states of `prices` over the leaf actions whose center
    /// preconditions hold in `center`. Lowers `prices` in place and returns
    /// the step that set each improved price (`None` for unchanged states).
    ///
    /// Steps change only on strict improvement, so they form a forest even
    /// with zero-cost actions.
    pub(crate) fn close(&self, center: &[Value], prices: &mut [Price]) -> Vec<Step> {
        let actions: Vec<&LeafAction> = self
            .actions
            .iter()
            .filter(|a| a.center_pre.iter().all(|f| center[f.var] == f.value))
            .collect();
        let mut steps = vec![None; self.size];
        let mut open: BinaryHeap<Reverse<(Price, usize)>> = prices
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p != UNREACHED)
            .map(|(rank, &p)| Reverse((p, rank)))
            .collect();

        while let Some(Reverse((price, rank))) = open.pop() {
            if price > prices[rank] {
                continue;
            }

            for action in actions.iter().filter(|a| self.satisfies(rank, &a.part.pre)) {
                let target = self.apply(rank, &action.part.eff);
                let reached = price.saturating_add(action.cost);
                if reached < prices[target] {
                    prices[target] = reached;
                    steps[target] = Some((rank, action.op));
                    open.push(Reverse((reached, target)));
                }
            }
        }

        steps
    }

    /// Lowers `out[offsets[var] + value]` to the cheapest price in `prices`
    /// of a reached leaf state with `var = value`, for every leaf variable.
    pub(crate) fn lower_fact_prices(&self, prices: &Prices, offsets: &[usize], out: &mut [Price]) {
        for (rank, price) in prices.iter() {
            for (slot, &var) in self.vars.iter().enumerate() {
                let fact = &mut out[offsets[var] + self.value(rank, slot)];
                *fact = (*fact).min(price);
            }
        }
    }

    /// The cheapest price of a reached leaf state that satisfies the leaf
    /// goal, with that state's rank; `None` if no goal state is reached.
    pub(crate) fn cheapest_goal(&self, prices: &Prices) -> Option<(Price, usize)> {
        prices
            .iter()
            .filter(|&(rank, _)| self.satisfies(rank, &self.goal))
            .map(|(rank, p)| (p, rank))
            .min()
    }
}
