//! Enumerated leaves for the symbolic search: leaves small enough that a
//! price per leaf state is cheaper than diagrams
//! ([`crate::Config::symbolic_min_states`]).
//!
//! An enumerated leaf is the explicit search's compiled [`Leaf`], without
//! derived variables or leaf reduction (the symbolic search rejects both),
//! so ranking, leaf reachability, transfers and the goal bound are the
//! explicit search's own.

use alloc::vec::Vec;

use sas_plus::{Fact, OpId, Value, VarId};

use crate::decoupled::{
    Leaf, LeafAction, LeafEffect, LeafFact, LeafPart, Price, Step, UNREACHED, View,
};
use crate::search::simulation;

/// A leaf's part of an operator, as `(slot, value)` lists.
pub(super) struct LocalFacts {
    /// Preconditions on the leaf.
    pub(super) pre: Vec<(usize, usize)>,
    /// Effects on the leaf, at most one per slot.
    pub(super) eff: Vec<(usize, usize)>,
}

impl LocalFacts {
    /// The same facts as the explicit search's [`LeafPart`], with
    /// unconditional effects.
    pub(super) fn part(&self) -> LeafPart {
        let fact = |&(slot, value): &(usize, usize)| LeafFact { slot, value };
        LeafPart {
            pre: self.pre.iter().map(fact).collect(),
            eff: self
                .eff
                .iter()
                .map(|f| LeafEffect {
                    center_cond: Vec::new(),
                    cond: Vec::new(),
                    fact: fact(f),
                })
                .collect(),
        }
    }
}

/// An enumerated leaf.
pub(super) struct ExplicitLeaf {
    /// The compiled leaf: ranking, goal and leaf actions.
    pub(super) leaf: Leaf,
    /// The center preconditions of each of [`Leaf::actions`], as
    /// `(center slot, value)`, the form [`super::Active`] takes.
    pub(super) center_pres: Vec<Vec<(usize, Value)>>,
}

impl ExplicitLeaf {
    /// The leaf over task variables `vars` with `domains`, initial values
    /// `initial` and goal facts `goal` as `(slot, value)`.
    ///
    /// # Panics
    ///
    /// Panics if it has more than [`crate::MAX_LEAF_STATES`] states, which
    /// the caller has ruled out.
    pub(super) fn new(
        vars: Vec<VarId>,
        domains: Vec<usize>,
        initial: &[usize],
        goal: &[(usize, usize)],
    ) -> Self {
        let mut leaf = Leaf::new(vars, domains, initial).expect("an enumerable leaf");
        leaf.goal = goal
            .iter()
            .map(|&(slot, value)| LeafFact { slot, value })
            .collect();
        Self {
            leaf,
            center_pres: Vec::new(),
        }
    }

    /// Adds leaf action `op` with its center preconditions and its part on
    /// the leaf.
    pub(super) fn add_action(
        &mut self,
        op: OpId,
        cost: Price,
        center_pre: Vec<(usize, Value)>,
        facts: &LocalFacts,
    ) {
        let mut center_facts: Vec<Fact> =
            center_pre.iter().map(|&(s, v)| Fact::new(s, v)).collect();
        center_facts.sort_unstable_by_key(|f| (f.var, f.value));
        self.leaf.actions.push(LeafAction {
            op,
            cost,
            center_pre: center_facts,
            part: facts.part(),
        });
        self.center_pres.push(center_pre);
    }

    /// The view used for rank-level operations; the symbolic search checks
    /// center preconditions itself.
    fn view(&self) -> View<'_> {
        self.leaf.static_view()
    }

    /// Leaf reachability from `seeds` over the leaf actions `active`
    /// accepts, with their costs (or 0 if `costs` is off): the prices and,
    /// per leaf state, the step that last lowered it.
    pub(super) fn close(
        &self,
        seeds: &[Price],
        active: super::Active<'_>,
        costs: bool,
    ) -> (Vec<Price>, Vec<Step>) {
        let mut prices = seeds.to_vec();
        let steps = self
            .view()
            .close_by(&mut prices, costs, |a| active(&self.center_pres[a]));
        (prices, steps)
    }

    /// `prices` carried across a global action's `part`; `None` if no leaf
    /// state survives.
    pub(super) fn transfer(&self, part: &LeafPart, prices: &[Price]) -> Option<Vec<Price>> {
        let next = self.view().transfer(part, prices);
        next.iter().any(|&p| p != UNREACHED).then_some(next)
    }

    /// A leaf state of `prices` at `price` that `part` carries to `target`.
    pub(super) fn source(
        &self,
        part: &LeafPart,
        prices: &[Price],
        target: usize,
        price: Price,
    ) -> usize {
        self.view().transfer_source(part, prices, target, price)
    }

    /// The least price of a reached goal leaf state in `prices`.
    pub(super) fn goal_price(&self, prices: &[Price]) -> Option<Price> {
        self.cheapest_goal(prices).map(|x| prices[x])
    }

    /// The rank of a reached goal leaf state with the least price.
    pub(super) fn cheapest_goal(&self, prices: &[Price]) -> Option<usize> {
        let view = self.view();
        reached(prices)
            .filter(|&(x, _)| view.is_goal(x))
            .min_by_key(|&(_, p)| p)
            .map(|(x, _)| x)
    }

    /// Lowers `out[offsets[var] + value]` to the cheapest reached leaf
    /// state of `prices` with `var = value`, for every leaf variable.
    pub(super) fn lower_fact_prices(&self, prices: &[Price], offsets: &[usize], out: &mut [Price]) {
        self.view().lower_fact_prices(reached(prices), offsets, out);
    }

    /// The cheapest leaf-action cost from the initial leaf state to a goal
    /// leaf state, center preconditions ignored, global `parts` free.
    pub(super) fn cheapest_goal_cost(&self, parts: &[&LeafPart]) -> Option<Price> {
        simulation::cheapest_goal(&self.leaf, parts)
    }

    /// The explicit dominance slack of stored `t` over new `n`: the largest
    /// `t(x) - n(x)` over the leaf states `n` reaches; `None` if `t` does
    /// not reach one of them.
    pub(super) fn slack(t: &[Price], n: &[Price]) -> Option<i128> {
        let mut delta = i128::MIN;
        for (x, p) in reached(n) {
            if t[x] == UNREACHED {
                return None;
            }

            delta = delta.max(i128::from(t[x]) - i128::from(p));
        }

        Some(delta)
    }
}

/// The reached leaf states of dense `prices` as `(rank, price)`.
fn reached(prices: &[Price]) -> impl Iterator<Item = (usize, Price)> + '_ {
    prices
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, p)| p != UNREACHED)
}
