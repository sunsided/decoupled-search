//! A [`Task`] compiled against a [`Factoring`]: center slots, compiled leaves,
//! global actions, and the decoupled transition function.
//!
//! An operator is a *leaf action* of leaf `L` when all its effects lie in `L`
//! and all its preconditions in the center or `L`. Every other operator is a
//! *global action*: it is searched explicitly, while leaf actions are priced
//! by [`Leaf::close`].

mod leaf;
mod prices;

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{Fact, OpId, Task, Value, VarId};

use crate::heuristic::FactPrices;
use crate::{Error, Factoring, MAX_LEAF_STATES};
pub(crate) use leaf::{Leaf, Price, UNREACHED};
use leaf::{LeafAction, LeafFact, LeafPart};
pub(crate) use prices::Prices;

/// Where a task variable lives in the factoring.
#[derive(Clone, Copy, Debug)]
enum Owner {
    /// Center slot.
    Center(usize),
    /// Leaf index and slot within the leaf.
    Leaf(usize, usize),
}

/// A list of facts split by owner.
#[derive(Debug)]
struct Split {
    /// Facts on center variables; `var` is the center slot.
    center: Vec<Fact>,
    /// Facts on each leaf, indexed by leaf.
    leaves: Vec<Vec<LeafFact>>,
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
    /// Effects on the center; `var` is the center slot.
    center_eff: Vec<Fact>,
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

impl DecoupledState {
    /// Whether `self` is at least as good as `other` in every price. Both
    /// states must have the same center; unreached prices compare as infinite.
    pub(crate) fn prices_at_most(&self, other: &Self) -> bool {
        self.prices
            .iter()
            .zip(&other.prices)
            .all(|(a, b)| a.at_most(b))
    }
}

/// The task in decoupled form.
#[derive(Clone, Debug)]
pub(crate) struct DecoupledTask {
    /// Initial values of the center variables, indexed by center slot.
    initial_center: Vec<Value>,
    /// The task variable of each center slot.
    center_vars: Vec<VarId>,
    /// Index of each task variable's first value in a [`FactPrices`]
    /// vector, plus a final entry with the number of facts.
    fact_offsets: Vec<usize>,
    /// The goal restricted to the center; `var` is the center slot.
    center_goal: Vec<Fact>,
    /// The compiled leaves, in [`Factoring::leaves`] order.
    pub(crate) leaves: Vec<Leaf>,
    /// All operators that are not leaf actions.
    pub(crate) globals: Vec<GlobalAction>,
}

impl DecoupledTask {
    /// Compiles `task` against `factoring`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidTask`] if the task does not validate,
    /// [`Error::Axioms`] or [`Error::ConditionalEffects`] for unsupported
    /// features, and [`Error::FactoringMismatch`] or [`Error::LeafTooLarge`]
    /// if `factoring` was built for a different task.
    pub(crate) fn new(task: &Task, factoring: &Factoring) -> Result<Self, Error> {
        check_supported(task)?;
        let owners = owners(task, factoring)?;

        let mut leaves = factoring
            .leaves()
            .iter()
            .enumerate()
            .map(|(index, vars)| compile_leaf(task, vars, index))
            .collect::<Result<Vec<_>, _>>()?;
        let goal = split(&owners, leaves.len(), &task.goal);
        for (leaf, facts) in leaves.iter_mut().zip(goal.leaves) {
            leaf.goal = facts;
        }

        let mut globals = Vec::new();
        for (op, operator) in task.operators.iter().enumerate() {
            let pre = split(&owners, leaves.len(), &operator.pre);
            let eff_facts: Vec<Fact> = operator.eff.iter().map(|e| e.fact).collect();
            let eff = split(&owners, leaves.len(), &eff_facts);
            let cost = Price::from(operator.cost);
            match leaf_action_owner(&pre, &eff) {
                Some(leaf) => leaves[leaf].actions.push(LeafAction {
                    op,
                    cost,
                    center_pre: pre.center,
                    part: LeafPart {
                        pre: pre.leaves[leaf].clone(),
                        eff: eff.leaves[leaf].clone(),
                    },
                }),
                None => globals.push(global_action(op, cost, pre, eff)),
            }
        }

        Ok(Self {
            initial_center: factoring
                .center()
                .iter()
                .map(|&v| task.initial.0[v])
                .collect(),
            center_vars: factoring.center().to_vec(),
            fact_offsets: fact_offsets(task),
            center_goal: goal.center,
            leaves,
            globals,
        })
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
    fn successor(&self, index: usize, state: &DecoupledState) -> Option<DecoupledState> {
        let action = &self.globals[index];
        if !holds(&action.center_pre, &state.center) {
            return None;
        }

        let mut dense = Vec::with_capacity(self.leaves.len());
        for ((leaf, part), old) in self.leaves.iter().zip(&action.parts).zip(&state.prices) {
            let old = old.to_dense(leaf.size);
            let Some(part) = part else {
                dense.push(old);
                continue;
            };

            let next = leaf.transfer(part, &old);
            if next.iter().all(|&p| p == UNREACHED) {
                return None;
            }

            dense.push(next);
        }

        let mut center = state.center.clone();
        for f in &action.center_eff {
            center[f.var] = f.value;
        }

        Some(self.closed(center, dense))
    }

    /// The decoupled state with center `center` and every leaf's seed prices
    /// `dense` closed under leaf reachability for that center, then stored
    /// compactly.
    fn closed(&self, center: Vec<Value>, mut dense: Vec<Vec<Price>>) -> DecoupledState {
        let prices = self
            .leaves
            .iter()
            .zip(&mut dense)
            .map(|(leaf, prices)| {
                leaf.close(&center, prices);
                Prices::from_dense(prices)
            })
            .collect();

        DecoupledState { center, prices }
    }

    /// The cheapest price of every fact over the member states of `state`:
    /// 0 for the center facts that hold, the cheapest reached leaf state
    /// containing it for leaf facts.
    pub(crate) fn fact_prices(&self, state: &DecoupledState) -> FactPrices<'_> {
        let mut prices = vec![UNREACHED; *self.fact_offsets.last().expect("has a final entry")];
        for (&var, &value) in self.center_vars.iter().zip(&state.center) {
            prices[self.fact_offsets[var] + widen(value)] = 0;
        }

        for (leaf, leaf_prices) in self.leaves.iter().zip(&state.prices) {
            leaf.lower_fact_prices(leaf_prices, &self.fact_offsets, &mut prices);
        }

        FactPrices {
            offsets: &self.fact_offsets,
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

        self.leaves
            .iter()
            .zip(&state.prices)
            .try_fold(0, |sum: Price, (leaf, prices)| {
                leaf.cheapest_goal(prices)
                    .map(|(p, _)| sum.saturating_add(p))
            })
    }
}

/// Rejects tasks this crate cannot search yet.
///
/// # Errors
///
/// See [`DecoupledTask::new`].
fn check_supported(task: &Task) -> Result<(), Error> {
    task.validate().map_err(Error::InvalidTask)?;
    if !task.axioms.is_empty() || task.variables.iter().any(sas_plus::Variable::is_derived) {
        return Err(Error::Axioms);
    }

    match task
        .operators
        .iter()
        .position(|o| o.eff.iter().any(|e| !e.conditions.is_empty()))
    {
        Some(op) => Err(Error::ConditionalEffects { op }),
        None => Ok(()),
    }
}

/// The owner of every task variable.
///
/// # Errors
///
/// [`Error::FactoringMismatch`] if the factoring names a variable the task
/// lacks or leaves one of its variables out.
fn owners(task: &Task, factoring: &Factoring) -> Result<Vec<Owner>, Error> {
    let mut owners = vec![None; task.num_variables()];
    let center = factoring
        .center()
        .iter()
        .enumerate()
        .map(|(slot, &v)| (v, Owner::Center(slot)));
    let leaves = factoring
        .leaves()
        .iter()
        .enumerate()
        .flat_map(|(leaf, vars)| {
            vars.iter()
                .enumerate()
                .map(move |(slot, &v)| (v, Owner::Leaf(leaf, slot)))
        });
    for (var, owner) in center.chain(leaves) {
        *owners.get_mut(var).ok_or(Error::FactoringMismatch)? = Some(owner);
    }

    owners
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::FactoringMismatch)
}

/// Compiles the ranking and initial state of leaf `index` over `vars`;
/// goal and actions are filled in later.
///
/// # Errors
///
/// [`Error::LeafTooLarge`] if the leaf has more than [`MAX_LEAF_STATES`] states.
fn compile_leaf(task: &Task, vars: &[VarId], index: usize) -> Result<Leaf, Error> {
    let domains: Vec<usize> = vars
        .iter()
        .map(|&v| widen(task.variables[v].domain_size))
        .collect();
    let mut strides = Vec::with_capacity(vars.len());
    let mut size = 1_usize;
    for &d in &domains {
        strides.push(size);
        size = size
            .checked_mul(d)
            .filter(|&s| s <= MAX_LEAF_STATES)
            .ok_or(Error::LeafTooLarge { leaf: index })?;
    }

    let initial = vars
        .iter()
        .zip(&strides)
        .map(|(&v, &stride)| widen(task.initial.0[v]) * stride)
        .sum();

    Ok(Leaf {
        vars: vars.to_vec(),
        strides,
        domains,
        size,
        initial,
        goal: Vec::new(),
        actions: Vec::new(),
    })
}

/// The index of every variable's first value in a vector over all facts of
/// `task`, plus a final entry with the number of facts.
fn fact_offsets(task: &Task) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(task.num_variables() + 1);
    let mut next = 0;
    offsets.push(next);
    for v in &task.variables {
        next += widen(v.domain_size);
        offsets.push(next);
    }

    offsets
}

/// Splits `facts` into center facts and per-leaf facts.
fn split(owners: &[Owner], leaf_count: usize, facts: &[Fact]) -> Split {
    let mut out = Split {
        center: Vec::new(),
        leaves: vec![Vec::new(); leaf_count],
    };
    for f in facts {
        match owners[f.var] {
            Owner::Center(slot) => out.center.push(Fact::new(slot, f.value)),
            Owner::Leaf(leaf, slot) => out.leaves[leaf].push(LeafFact {
                slot,
                value: widen(f.value),
            }),
        }
    }

    out
}

/// The leaf an operator is a leaf action of, given its split preconditions
/// and effects: the only leaf it touches, provided it sets no center variable.
fn leaf_action_owner(pre: &Split, eff: &Split) -> Option<usize> {
    if !eff.center.is_empty() {
        return None;
    }

    let mut touched =
        (0..eff.leaves.len()).filter(|&l| !pre.leaves[l].is_empty() || !eff.leaves[l].is_empty());
    let leaf = touched.next()?;
    (touched.next().is_none() && !eff.leaves[leaf].is_empty()).then_some(leaf)
}

/// Builds a global action from split preconditions and effects.
fn global_action(op: OpId, cost: Price, pre: Split, eff: Split) -> GlobalAction {
    let parts = pre
        .leaves
        .into_iter()
        .zip(eff.leaves)
        .map(|(pre, eff)| (!pre.is_empty() || !eff.is_empty()).then_some(LeafPart { pre, eff }))
        .collect();

    GlobalAction {
        op,
        cost,
        center_pre: pre.center,
        center_eff: eff.center,
        parts,
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
