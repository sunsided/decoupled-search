//! One compiled leaf factor: ranking of leaf states and leaf reachability.
//!
//! A leaf state is a mixed-radix rank over the leaf's state variables. Its
//! *slots* are those variables, followed by the derived variables attached
//! to the leaf, whose values depend on the rank and the current center. A
//! [`View`] fixes the center, evaluates the derived slots once for every
//! rank, and offers all rank-level operations.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Fact, OpId, Value, VarId};

use super::Prices;
use super::axioms::{Axioms, Scope};

/// The cost of reaching a leaf state; [`UNREACHED`] when it is not reached.
pub(crate) type Price = u64;

/// The price of a leaf state that is not reached.
pub(crate) const UNREACHED: Price = Price::MAX;

/// How a leaf state was first reached at its final price during a closure:
/// from leaf state `from` (a rank) by leaf action `op`.
pub(crate) type Step = Option<(usize, OpId)>;

/// A condition or assignment on one leaf slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LeafFact {
    /// Position in the leaf: a variable of [`crate::Factoring::leaves`]
    /// below [`Leaf::vars`]`.len()`, an attached derived variable above.
    pub(crate) slot: usize,
    /// The value, already widened for rank arithmetic.
    pub(crate) value: usize,
}

/// One effect of an operator on a leaf variable, possibly conditional.
#[derive(Clone, Debug)]
pub(crate) struct LeafEffect {
    /// Effect conditions on the center; `var` is the center slot. Evaluated
    /// on the center the operator is applied in.
    pub(crate) center_cond: Vec<Fact>,
    /// Effect conditions on this leaf, evaluated on the leaf state the
    /// operator is applied in.
    pub(crate) cond: Vec<LeafFact>,
    /// The assignment; its slot is always a state variable of the leaf.
    pub(crate) fact: LeafFact,
}

impl LeafEffect {
    /// Whether the effect has conditions on the center.
    pub(crate) fn reads_center(&self) -> bool {
        !self.center_cond.is_empty()
    }
}

/// The part of an operator that reads or writes one leaf.
#[derive(Clone, Debug)]
pub(crate) struct LeafPart {
    /// Preconditions on this leaf's slots.
    pub(crate) pre: Vec<LeafFact>,
    /// Effects on this leaf's variables, in operator order: a later firing
    /// effect on the same variable wins.
    pub(crate) eff: Vec<LeafEffect>,
}

impl LeafPart {
    /// Whether any effect depends on the center.
    pub(crate) fn reads_center(&self) -> bool {
        self.eff.iter().any(LeafEffect::reads_center)
    }
}

/// An operator whose effects all lie in one leaf and whose preconditions and
/// effect conditions lie in the center and that leaf.
#[derive(Clone, Debug)]
pub(crate) struct LeafAction {
    /// The operator in the original task.
    pub(crate) op: OpId,
    /// The operator cost.
    pub(crate) cost: Price,
    /// Preconditions on the center; `var` is the center slot, not the task
    /// variable. Sorted, so that subset tests are linear.
    pub(crate) center_pre: Vec<Fact>,
    /// Preconditions and effects on the leaf.
    pub(crate) part: LeafPart,
}

/// A leaf factor whose states are mixed-radix ranks over its variables.
#[derive(Clone, Debug)]
pub(crate) struct Leaf {
    /// The task variable of each state-variable slot.
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
    /// The derived variables attached to this leaf, by slot after
    /// [`Leaf::vars`], with the rules that evaluate them.
    pub(crate) derived: Scope,
    /// The task goal restricted to this leaf.
    pub(crate) goal: Vec<LeafFact>,
    /// All leaf actions of this leaf.
    pub(crate) actions: Vec<LeafAction>,
    /// Leaf transitions removed by simulation-based leaf reduction, by
    /// `rank * actions.len() + action`; empty if none are. Leaf closure
    /// skips them.
    pub(crate) disabled: Vec<bool>,
}

impl Leaf {
    /// Price vector with only the initial leaf state reached, at price 0.
    pub(crate) fn initial_prices(&self) -> Vec<Price> {
        let mut prices = vec![UNREACHED; self.size];
        prices[self.initial] = 0;
        prices
    }

    /// The value of state-variable slot `slot` in leaf state `rank`.
    pub(crate) fn var_value(&self, rank: usize, slot: usize) -> usize {
        rank / self.strides[slot] % self.domains[slot]
    }

    /// Leaf state `rank` with state-variable slot `slot` set to `value`.
    fn with_value(&self, rank: usize, slot: usize, value: usize) -> usize {
        rank - self.var_value(rank, slot) * self.strides[slot] + value * self.strides[slot]
    }

    /// The view of this leaf under center state `center` (values by center
    /// slot, as in [`crate::Factoring::center`] order without attached
    /// derived variables). Evaluates the attached derived variables for
    /// every rank, which costs `O(size * rules)` if there are any.
    pub(crate) fn view<'a>(&'a self, axioms: &Axioms, center: &'a [Value]) -> View<'a> {
        let derived = if self.derived.is_empty() {
            Vec::new()
        } else {
            self.derived_values(axioms, center)
        };

        View {
            leaf: self,
            center,
            derived,
        }
    }

    /// The view of a leaf that does not read the center (see
    /// [`Leaf::reads_center`]), valid under every center. Center
    /// preconditions of leaf actions are not checked by its rank-level
    /// operations, only by [`View::close`], which must not be called on it.
    pub(crate) fn static_view(&self) -> View<'_> {
        debug_assert!(self.derived.is_empty(), "derived slots read the center");
        View {
            leaf: self,
            center: &[],
            derived: Vec::new(),
        }
    }

    /// Whether the leaf's behaviour depends on the center beyond the center
    /// preconditions of its leaf actions: an effect on it (of a leaf action
    /// or of a global action in `parts`) has center conditions, or a derived
    /// variable is attached to it. Dominance analyses that hold "for every
    /// center" fall back to plain price comparison on such leaves.
    pub(crate) fn reads_center<'p>(&self, parts: impl IntoIterator<Item = &'p LeafPart>) -> bool {
        !self.derived.is_empty()
            || self.actions.iter().any(|a| a.part.reads_center())
            || parts.into_iter().any(LeafPart::reads_center)
    }

    /// The values of the attached derived variables for every rank under
    /// `center`, rank-major.
    fn derived_values(&self, axioms: &Axioms, center: &[Value]) -> Vec<usize> {
        let mut scratch = axioms.scratch(center);
        let mut out = Vec::with_capacity(self.size * self.derived.vars.len());
        for rank in 0..self.size {
            for (slot, &var) in self.vars.iter().enumerate() {
                scratch[var] = u32::try_from(self.var_value(rank, slot)).expect("values fit u32");
            }

            axioms.evaluate(&self.derived, &mut scratch);
            out.extend(self.derived.vars.iter().map(|&v| super::widen(scratch[v])));
        }

        out
    }
}

/// A [`Leaf`] under a fixed center state.
#[derive(Debug)]
pub(crate) struct View<'a> {
    /// The leaf.
    pub(crate) leaf: &'a Leaf,
    /// The center state, by center slot.
    center: &'a [Value],
    /// The value of every attached derived variable in every rank,
    /// rank-major; empty if the leaf has none.
    derived: Vec<usize>,
}

impl View<'_> {
    /// The value of slot `slot` in leaf state `rank`.
    fn value(&self, rank: usize, slot: usize) -> usize {
        let vars = self.leaf.vars.len();
        if slot < vars {
            self.leaf.var_value(rank, slot)
        } else {
            self.derived[rank * self.leaf.derived.vars.len() + slot - vars]
        }
    }

    /// Whether leaf state `rank` satisfies all `facts`.
    pub(crate) fn satisfies(&self, rank: usize, facts: &[LeafFact]) -> bool {
        facts.iter().all(|f| self.value(rank, f.slot) == f.value)
    }

    /// Whether all center `facts` (by center slot) hold.
    fn center_holds(&self, facts: &[Fact]) -> bool {
        facts.iter().all(|f| self.center[f.var] == f.value)
    }

    /// The leaf state after applying the effects `eff` to `rank`: every
    /// effect whose conditions hold in `rank` and the center fires, in
    /// order.
    pub(crate) fn apply(&self, rank: usize, eff: &[LeafEffect]) -> usize {
        eff.iter()
            .filter(|e| self.center_holds(&e.center_cond) && self.satisfies(rank, &e.cond))
            .fold(rank, |r, e| {
                self.leaf.with_value(r, e.fact.slot, e.fact.value)
            })
    }

    /// The prices after a global action with leaf part `part`, applied under
    /// this view's center: reached states that satisfy its preconditions,
    /// mapped through its effects, keeping the cheapest price when two states
    /// map to the same one.
    pub(crate) fn transfer(&self, part: &LeafPart, prices: &[Price]) -> Vec<Price> {
        let mut next = vec![UNREACHED; self.leaf.size];
        for (rank, &price) in prices.iter().enumerate() {
            if price == UNREACHED || !self.satisfies(rank, &part.pre) {
                continue;
            }

            let target = self.apply(rank, &part.eff);
            next[target] = next[target].min(price);
        }

        next
    }

    /// A reached leaf state of `prices` that [`View::transfer`] maps onto
    /// `seed` at exactly `price`.
    ///
    /// # Panics
    ///
    /// Panics if there is none, which means `price` did not come from a
    /// transfer of `prices` under this view.
    pub(crate) fn transfer_source(
        &self,
        part: &LeafPart,
        prices: &[Price],
        seed: usize,
        price: Price,
    ) -> usize {
        (0..self.leaf.size)
            .find(|&rank| {
                prices[rank] == price
                    && self.satisfies(rank, &part.pre)
                    && self.apply(rank, &part.eff) == seed
            })
            .expect("a transferred price has a source leaf state")
    }

    /// Leaf reachability under the view's center: Dijkstra from all reached
    /// states of `prices` over the leaf actions whose center preconditions
    /// hold. Lowers `prices` in place and returns the step that set each
    /// improved price (`None` for unchanged states).
    ///
    /// Steps change only on strict improvement, so they form a forest even
    /// with zero-cost actions. Without `costs`, every action counts as free,
    /// so only reachability is recorded.
    pub(crate) fn close(&self, prices: &mut [Price], costs: bool) -> Vec<Step> {
        let count = self.leaf.actions.len();
        let actions: Vec<(usize, &LeafAction)> = self
            .leaf
            .actions
            .iter()
            .enumerate()
            .filter(|(_, a)| self.center_holds(&a.center_pre))
            .collect();
        let disabled = |rank: usize, index: usize| {
            self.leaf
                .disabled
                .get(rank * count + index)
                .is_some_and(|&d| d)
        };
        let mut steps = vec![None; self.leaf.size];
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

            for &(_, action) in actions
                .iter()
                .filter(|&&(i, a)| !disabled(rank, i) && self.satisfies(rank, &a.part.pre))
            {
                let target = self.apply(rank, &action.part.eff);
                let cost = if costs { action.cost } else { 0 };
                let reached = price.saturating_add(cost);
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
    /// of a reached leaf state with `var = value`, for every leaf variable
    /// and attached derived variable.
    pub(crate) fn lower_fact_prices(&self, prices: &Prices, offsets: &[usize], out: &mut [Price]) {
        let slots = self.leaf.vars.iter().chain(&self.leaf.derived.vars);
        for (rank, price) in prices.iter() {
            for (slot, &var) in slots.clone().enumerate() {
                let fact = &mut out[offsets[var] + self.value(rank, slot)];
                *fact = (*fact).min(price);
            }
        }
    }

    /// Whether leaf state `rank` satisfies the leaf goal.
    pub(crate) fn is_goal(&self, rank: usize) -> bool {
        self.satisfies(rank, &self.leaf.goal)
    }

    /// The cheapest price of a reached leaf state that satisfies the leaf
    /// goal, with that state's rank; `None` if no goal state is reached.
    pub(crate) fn cheapest_goal(&self, prices: &Prices) -> Option<(Price, usize)> {
        prices
            .iter()
            .filter(|&(rank, _)| self.is_goal(rank))
            .map(|(rank, p)| (p, rank))
            .min()
    }
}
