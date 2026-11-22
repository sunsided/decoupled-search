//! A [`Task`] compiled against a [`Factoring`]: center slots, compiled leaves,
//! global actions, and the decoupled transition function.
//!
//! An operator is a *leaf action* of leaf `L` when all its effects lie in `L`
//! and all its preconditions and effect conditions in the center or `L`.
//! Every other operator is a *global action*: it is searched explicitly,
//! while leaf actions are priced by [`leaf::View::close`].
//!
//! Conditional effects are evaluated per leaf state where they can be: an
//! effect on leaf `L` may be conditioned on the center and `L`, an effect on
//! the center only on the center. Derived variables are evaluated in the
//! factor they depend on (see [`axioms`]).

mod axioms;
mod compile;
mod leaf;
mod prices;

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{Fact, OpId, Task, Value, VarId};

use crate::heuristic::FactPrices;
use crate::{Error, Factoring};
use axioms::{Axioms, Scope};
pub(crate) use compile::{leaf_action_of, mobile_leaves};
pub(crate) use leaf::{Leaf, LeafAction, LeafFact, LeafPart, Price, UNREACHED, View};
pub(crate) use prices::Prices;

/// One effect of a global action on a center variable.
#[derive(Clone, Debug)]
pub(crate) struct CenterEffect {
    /// Effect conditions; `var` is the center slot. Evaluated on the center
    /// the action is applied in.
    cond: Vec<Fact>,
    /// The assignment; `var` is the center slot.
    fact: Fact,
}

/// An operator that is not a leaf action.
#[derive(Clone, Debug)]
pub(crate) struct GlobalAction {
    /// The operator in the original task.
    pub(crate) op: OpId,
    /// The operator cost.
    pub(crate) cost: Price,
    /// Preconditions on the center; `var` is the center slot.
    center_pre: Vec<Fact>,
    /// Effects on the center, in operator order.
    center_eff: Vec<CenterEffect>,
    /// The operator's part on each leaf, `None` for leaves it does not touch.
    pub(crate) parts: Vec<Option<LeafPart>>,
}

/// A center state together with the price of every leaf state of every leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecoupledState {
    /// Values of the center variables, indexed by center slot.
    pub(crate) center: Vec<Value>,
    /// The reached leaf states of each leaf with their prices, indexed by
    /// leaf. Never empty for a leaf: its initial or seed state is reached.
    pub(crate) prices: Vec<Prices>,
}

/// The task in decoupled form.
#[derive(Clone, Debug)]
pub(crate) struct DecoupledTask {
    /// Initial values of the center variables, indexed by center slot, with
    /// derived center variables evaluated.
    initial_center: Vec<Value>,
    /// The task variable of each center slot: the factoring's center minus
    /// the derived variables attached to a leaf.
    pub(crate) center_vars: Vec<VarId>,
    /// The derived center variables and their rules.
    center_derived: Scope,
    /// The task's axioms, for derived center and leaf variables.
    pub(crate) axioms: Axioms,
    /// Index of each task variable's first value in a [`FactPrices`]
    /// vector, plus a final entry with the number of facts.
    fact_offsets: Vec<usize>,
    /// The leaf of each task variable (derived ones included), `None` for
    /// center variables.
    leaf_of: Vec<Option<usize>>,
    /// The goal restricted to the center; `var` is the center slot.
    center_goal: Vec<Fact>,
    /// The compiled leaves, in [`Factoring::leaves`] order.
    pub(crate) leaves: Vec<Leaf>,
    /// All operators that are not leaf actions.
    pub(crate) globals: Vec<GlobalAction>,
    /// Whether leaf action costs are ignored during leaf closure, so that
    /// prices only record reachability (every reached leaf state at 0).
    reachability_only: bool,
}

impl DecoupledTask {
    /// Compiles `task` against `factoring`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidTask`] if the task does not validate; one of
    /// [`Error::CenterEffectReadsLeaf`], [`Error::EffectConditionAcrossLeaves`],
    /// [`Error::AxiomAcrossLeaves`] or [`Error::DerivedInLeaf`] for
    /// conditional effects and axioms the decoupled semantics cannot
    /// express; and [`Error::FactoringMismatch`] or [`Error::LeafTooLarge`]
    /// if `factoring` was built for a different task.
    pub(crate) fn new(task: &Task, factoring: &Factoring) -> Result<Self, Error> {
        compile::compile(task, factoring)
    }

    /// The same task with leaf prices replaced by plain reachability: every
    /// reached leaf state has price 0, and leaf action costs only enter the
    /// plan, not the search.
    pub(crate) fn into_reachability_only(mut self) -> Self {
        self.reachability_only = true;
        self
    }

    /// Whether prices only record reachability.
    pub(crate) fn is_reachability_only(&self) -> bool {
        self.reachability_only
    }

    /// The view of leaf `leaf` under center state `center`.
    pub(crate) fn view<'a>(&'a self, leaf: usize, center: &'a [Value]) -> View<'a> {
        self.leaves[leaf].view(&self.axioms, center)
    }

    /// The initial decoupled state: the initial center, and every leaf closed
    /// from its initial leaf state at price 0.
    pub(crate) fn initial(&self) -> DecoupledState {
        let dense = self.leaves.iter().map(Leaf::initial_prices).collect();
        self.closed(self.initial_center.clone(), dense)
    }

    /// All successors of `state` as `(global action index, successor)`.
    pub(crate) fn successors(&self, state: &DecoupledState) -> Vec<(usize, DecoupledState)> {
        (0..self.globals.len())
            .filter_map(|i| self.successor(i, state).map(|s| (i, s)))
            .collect()
    }

    /// The successor of `state` under global action `index`, or `None` if it
    /// is not applicable.
    pub(crate) fn successor(&self, index: usize, state: &DecoupledState) -> Option<DecoupledState> {
        let action = &self.globals[index];
        if !holds(&action.center_pre, &state.center) {
            return None;
        }

        let dense = self.transferred(action, state)?;
        Some(self.closed(self.next_center(action, &state.center), dense))
    }

    /// The dense prices of every leaf of `state` carried across `action`
    /// (unchanged for leaves it does not touch); `None` if a leaf it touches
    /// has no reached leaf state satisfying its preconditions.
    pub(crate) fn transferred(
        &self,
        action: &GlobalAction,
        state: &DecoupledState,
    ) -> Option<Vec<Vec<Price>>> {
        let mut dense = Vec::with_capacity(self.leaves.len());
        for (leaf, (part, old)) in action.parts.iter().zip(&state.prices).enumerate() {
            let old = old.to_dense(self.leaves[leaf].size);
            let Some(part) = part else {
                dense.push(old);
                continue;
            };

            let next = self.view(leaf, &state.center).transfer(part, &old);
            if next.iter().all(|&p| p == UNREACHED) {
                return None;
            }

            dense.push(next);
        }

        Some(dense)
    }

    /// The center after applying `action` to `center`: firing effects in
    /// order, conditions read from `center`, then derived center variables
    /// re-evaluated.
    fn next_center(&self, action: &GlobalAction, center: &[Value]) -> Vec<Value> {
        let mut next = center.to_vec();
        for e in action.center_eff.iter().filter(|e| holds(&e.cond, center)) {
            next[e.fact.var] = e.fact.value;
        }

        self.axioms.evaluate_center(&self.center_derived, &mut next);
        next
    }

    /// The decoupled state with center `center` and every leaf's seed prices
    /// `dense` closed under leaf reachability for that center, then stored
    /// compactly.
    fn closed(&self, center: Vec<Value>, mut dense: Vec<Vec<Price>>) -> DecoupledState {
        let prices = dense
            .iter_mut()
            .enumerate()
            .map(|(leaf, prices)| {
                self.close(leaf, &center, prices);
                Prices::from_dense(prices)
            })
            .collect();

        DecoupledState { center, prices }
    }

    /// Leaf reachability of leaf `leaf` under `center` from the seed
    /// `prices`, lowered in place; returns the closure steps. In
    /// reachability-only mode, leaf action costs count as 0.
    pub(crate) fn close(
        &self,
        leaf: usize,
        center: &[Value],
        prices: &mut [Price],
    ) -> Vec<leaf::Step> {
        self.view(leaf, center)
            .close(prices, !self.reachability_only)
    }

    /// The cheapest price of every fact over the member states of `state`:
    /// 0 for the center facts that hold, the cheapest reached leaf state
    /// containing it for leaf facts (derived ones included).
    pub(crate) fn fact_prices(&self, state: &DecoupledState) -> FactPrices<'_> {
        let mut prices = vec![UNREACHED; *self.fact_offsets.last().expect("has a final entry")];
        for (&var, &value) in self.center_vars.iter().zip(&state.center) {
            prices[self.fact_offsets[var] + widen(value)] = 0;
        }

        for (leaf, leaf_prices) in state.prices.iter().enumerate() {
            self.view(leaf, &state.center).lower_fact_prices(
                leaf_prices,
                &self.fact_offsets,
                &mut prices,
            );
        }

        FactPrices {
            offsets: Cow::Borrowed(&self.fact_offsets),
            leaves: Cow::Borrowed(&self.leaf_of),
            prices,
        }
    }

    /// The cost still to pay in the leaves to reach the goal from `state`:
    /// the sum of the cheapest goal prices, or `None` if `state` contains no
    /// goal state.
    pub(crate) fn goal_price(&self, state: &DecoupledState) -> Option<Price> {
        if !holds(&self.center_goal, &state.center) {
            return None;
        }

        state
            .prices
            .iter()
            .enumerate()
            .try_fold(0, |sum: Price, (leaf, prices)| {
                self.view(leaf, &state.center)
                    .cheapest_goal(prices)
                    .map(|(p, _)| sum.saturating_add(p))
            })
    }
}

#[cfg(test)]
impl DecoupledTask {
    /// Every member state of `state` with its price (the sum of its leaf
    /// states' prices), derived variables evaluated. Exponential in the
    /// number of leaves; for tests on tiny tasks.
    pub(crate) fn member_states(
        &self,
        task: &Task,
        state: &DecoupledState,
    ) -> Vec<(sas_plus::State, Price)> {
        let mut members: Vec<(sas_plus::State, Price)> = vec![(task.initial.clone(), 0)];
        for (slot, &var) in self.center_vars.iter().enumerate() {
            members[0].0.0[var] = state.center[slot];
        }

        for (leaf, prices) in self.leaves.iter().zip(&state.prices) {
            members = members
                .into_iter()
                .flat_map(|(member, price)| {
                    prices.iter().map(move |(rank, p)| {
                        let mut member = member.clone();
                        for (slot, &var) in leaf.vars.iter().enumerate() {
                            let value = leaf.var_value(rank, slot);
                            member.0[var] = u32::try_from(value).expect("values fit u32");
                        }

                        (member, price.saturating_add(p))
                    })
                })
                .collect();
        }

        for (member, _) in &mut members {
            task.evaluate_axioms(member);
        }

        members
    }
}

/// Whether all `facts` (with `var` as center slot) hold in `center`.
fn holds(facts: &[Fact], center: &[Value]) -> bool {
    facts.iter().all(|f| center[f.var] == f.value)
}

/// Widens a domain value or size for rank arithmetic.
fn widen(value: Value) -> usize {
    usize::try_from(value).expect("a u32 fits in usize on supported targets")
}
