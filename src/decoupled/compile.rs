//! Compiles a [`Task`] against a [`Factoring`] into a [`DecoupledTask`].
//!
//! ```mermaid
//! graph LR
//!     check[validate task] --> owners[owner of every variable]
//!     owners --> derived[attach derived variables to center or one leaf]
//!     derived --> leaves[rank leaves, split goal]
//!     leaves --> ops[split every operator: leaf action or global action]
//! ```

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{Effect, Fact, OpId, Task, VarId};

use super::axioms::Axioms;
use super::leaf::LeafEffect;
use super::{
    CenterEffect, DecoupledTask, GlobalAction, Leaf, LeafAction, LeafFact, LeafPart, Price, widen,
};
use crate::heuristic::fact_offsets;
use crate::{Error, Factoring, MAX_LEAF_STATES};

/// Where a task variable lives in the decoupled task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    /// Center slot.
    Center(usize),
    /// Leaf index and slot within the leaf.
    Leaf(usize, usize),
}

impl Owner {
    /// The leaf index, `None` for the center.
    fn leaf(self) -> Option<usize> {
        match self {
            Self::Center(_) => None,
            Self::Leaf(leaf, _) => Some(leaf),
        }
    }
}

/// A list of facts split by owner.
#[derive(Debug)]
struct Split {
    /// Facts on center variables; `var` is the center slot.
    center: Vec<Fact>,
    /// Facts on each leaf, indexed by leaf.
    leaves: Vec<Vec<LeafFact>>,
}

/// The effects of one operator split by owner.
#[derive(Debug)]
struct SplitEffects {
    /// Effects on center variables.
    center: Vec<CenterEffect>,
    /// Effects on each leaf, indexed by leaf.
    leaves: Vec<Vec<LeafEffect>>,
    /// The leaves the effect conditions read, indexed by leaf.
    reads: Vec<bool>,
}

/// The ownership of every task variable.
struct Ownership {
    /// The owner of every task variable.
    owners: Vec<Owner>,
    /// The task variable of each center slot.
    center_vars: Vec<VarId>,
    /// The derived center variables.
    center_derived: Vec<VarId>,
    /// The derived variables attached to each leaf.
    leaf_derived: Vec<Vec<VarId>>,
}

/// Compiles `task` against `factoring`.
///
/// # Errors
///
/// See [`DecoupledTask::new`].
pub(super) fn compile(task: &Task, factoring: &Factoring) -> Result<DecoupledTask, Error> {
    task.validate().map_err(Error::InvalidTask)?;
    let ownership = ownership(task, factoring)?;
    let axioms = Axioms::new(task, ownership.center_vars.clone());
    let leaf_count = factoring.leaves().len();

    let mut leaves = factoring
        .leaves()
        .iter()
        .zip(&ownership.leaf_derived)
        .enumerate()
        .map(|(index, (vars, derived))| {
            compile_leaf(task, vars, index, axioms.scope(derived.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let goal = split(&ownership.owners, leaf_count, &task.goal);
    for (leaf, facts) in leaves.iter_mut().zip(goal.leaves) {
        leaf.goal = facts;
    }

    let mut globals = Vec::new();
    for (op, operator) in task.operators.iter().enumerate() {
        let pre = split(&ownership.owners, leaf_count, &operator.pre);
        let eff = split_effects(&ownership.owners, leaf_count, op, &operator.eff)?;
        let cost = Price::from(operator.cost);
        match leaf_action_owner(&pre, &eff) {
            Some(leaf) => leaves[leaf]
                .actions
                .push(leaf_action(op, cost, pre, eff, leaf)),
            None => globals.push(global_action(op, cost, pre, eff)),
        }
    }

    let initial = task.initial_state();
    Ok(DecoupledTask {
        initial_center: ownership
            .center_vars
            .iter()
            .map(|&v| initial.0[v])
            .collect(),
        center_derived: axioms.scope(ownership.center_derived),
        center_vars: ownership.center_vars,
        axioms,
        fact_offsets: fact_offsets(task),
        leaf_of: ownership.owners.iter().map(|o| o.leaf()).collect(),
        center_goal: goal.center,
        leaves,
        globals,
        reachability_only: false,
    })
}

/// The owner of every task variable: the factoring's leaves own their
/// variables; a derived variable joins the one leaf it depends on, if any,
/// and the center otherwise.
///
/// # Errors
///
/// [`Error::FactoringMismatch`] if the factoring does not partition the
/// task's variables, [`Error::DerivedInLeaf`] if a leaf names a derived
/// variable, and [`Error::AxiomAcrossLeaves`] if a derived variable depends
/// on two leaves.
fn ownership(task: &Task, factoring: &Factoring) -> Result<Ownership, Error> {
    let leaf_of = factoring.leaf_of(task)?;
    if let Some(var) =
        (0..task.num_variables()).find(|&v| leaf_of[v].is_some() && task.variables[v].is_derived())
    {
        return Err(Error::DerivedInLeaf { var });
    }

    let axioms = Axioms::new(task, Vec::new());
    let mut out = Ownership {
        owners: vec![Owner::Center(0); task.num_variables()],
        center_vars: Vec::new(),
        center_derived: Vec::new(),
        leaf_derived: vec![Vec::new(); factoring.leaves().len()],
    };
    for (leaf, vars) in factoring.leaves().iter().enumerate() {
        for (slot, &var) in vars.iter().enumerate() {
            out.owners[var] = Owner::Leaf(leaf, slot);
        }
    }

    for &var in factoring.center() {
        if !task.variables[var].is_derived() {
            out.owners[var] = Owner::Center(out.center_vars.len());
            out.center_vars.push(var);
            continue;
        }

        if let Some(leaf) = derived_leaf(task, &axioms, &leaf_of, var)? {
            let slot = factoring.leaves()[leaf].len() + out.leaf_derived[leaf].len();
            out.owners[var] = Owner::Leaf(leaf, slot);
            out.leaf_derived[leaf].push(var);
        } else {
            out.owners[var] = Owner::Center(out.center_vars.len());
            out.center_vars.push(var);
            out.center_derived.push(var);
        }
    }

    Ok(out)
}

/// The one leaf whose state variables derived variable `var` depends on,
/// directly or through other derived variables; `None` if it depends only
/// on the center.
///
/// # Errors
///
/// [`Error::AxiomAcrossLeaves`] if it depends on two leaves.
fn derived_leaf(
    task: &Task,
    axioms: &Axioms,
    leaf_of: &[Option<usize>],
    var: VarId,
) -> Result<Option<usize>, Error> {
    let mut seen = vec![false; task.num_variables()];
    let mut stack = vec![var];
    seen[var] = true;
    let mut found = None;
    while let Some(v) = stack.pop() {
        for dep in axioms.dependencies(v) {
            if core::mem::replace(&mut seen[dep], true) {
                continue;
            }

            if task.variables[dep].is_derived() {
                stack.push(dep);
                continue;
            }

            let Some(leaf) = leaf_of[dep] else {
                continue;
            };

            match found {
                None => found = Some(leaf),
                Some(other) if other != leaf => {
                    return Err(Error::AxiomAcrossLeaves {
                        var,
                        leaves: (other.min(leaf), other.max(leaf)),
                    });
                }
                Some(_) => {}
            }
        }
    }

    Ok(found)
}

/// Compiles the ranking and initial state of leaf `index` over `vars`, with
/// the attached derived variables of `derived`; goal and actions are filled
/// in later.
///
/// # Errors
///
/// [`Error::LeafTooLarge`] if the leaf has more than [`MAX_LEAF_STATES`] states.
fn compile_leaf(
    task: &Task,
    vars: &[VarId],
    index: usize,
    derived: super::Scope,
) -> Result<Leaf, Error> {
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
        derived,
        goal: Vec::new(),
        actions: Vec::new(),
    })
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

    out.center.sort_unstable_by_key(|f| (f.var, f.value));
    out
}

/// Splits the effects of operator `op` by owner.
///
/// # Errors
///
/// [`Error::CenterEffectReadsLeaf`] if an effect on the center is
/// conditioned on a leaf, and [`Error::EffectConditionAcrossLeaves`] if an
/// effect on one leaf is conditioned on another.
fn split_effects(
    owners: &[Owner],
    leaf_count: usize,
    op: OpId,
    effects: &[Effect],
) -> Result<SplitEffects, Error> {
    let mut out = SplitEffects {
        center: Vec::new(),
        leaves: vec![Vec::new(); leaf_count],
        reads: vec![false; leaf_count],
    };
    for (effect, e) in effects.iter().enumerate() {
        let cond = split(owners, leaf_count, &e.conditions);
        let read: Vec<usize> = (0..leaf_count)
            .filter(|&l| !cond.leaves[l].is_empty())
            .collect();
        for &l in &read {
            out.reads[l] = true;
        }

        match owners[e.fact.var] {
            Owner::Center(slot) => {
                if !read.is_empty() {
                    return Err(Error::CenterEffectReadsLeaf { op, effect });
                }

                out.center.push(CenterEffect {
                    cond: cond.center,
                    fact: Fact::new(slot, e.fact.value),
                });
            }
            Owner::Leaf(leaf, slot) => {
                if read.iter().any(|&l| l != leaf) {
                    return Err(Error::EffectConditionAcrossLeaves { op, effect });
                }

                out.leaves[leaf].push(LeafEffect {
                    center_cond: cond.center,
                    cond: cond.leaves.into_iter().nth(leaf).unwrap_or_default(),
                    fact: LeafFact {
                        slot,
                        value: widen(e.fact.value),
                    },
                });
            }
        }
    }

    Ok(out)
}

/// The leaf an operator is a leaf action of, given its split preconditions
/// and effects: the only leaf it touches, provided it sets no center
/// variable and sets something in that leaf.
fn leaf_action_owner(pre: &Split, eff: &SplitEffects) -> Option<usize> {
    if !eff.center.is_empty() {
        return None;
    }

    let mut touched = (0..eff.leaves.len()).filter(|&l| touches(pre, eff, l));
    let leaf = touched.next()?;
    (touched.next().is_none() && !eff.leaves[leaf].is_empty()).then_some(leaf)
}

/// Whether an operator reads or writes leaf `leaf`.
fn touches(pre: &Split, eff: &SplitEffects, leaf: usize) -> bool {
    !pre.leaves[leaf].is_empty() || !eff.leaves[leaf].is_empty() || eff.reads[leaf]
}

/// Builds the leaf action of leaf `leaf` from split preconditions and effects.
fn leaf_action(op: OpId, cost: Price, pre: Split, eff: SplitEffects, leaf: usize) -> LeafAction {
    LeafAction {
        op,
        cost,
        center_pre: pre.center,
        part: LeafPart {
            pre: pre.leaves.into_iter().nth(leaf).unwrap_or_default(),
            eff: eff.leaves.into_iter().nth(leaf).unwrap_or_default(),
        },
    }
}

/// Builds a global action from split preconditions and effects.
fn global_action(op: OpId, cost: Price, pre: Split, eff: SplitEffects) -> GlobalAction {
    let parts = pre
        .leaves
        .into_iter()
        .zip(eff.leaves)
        .zip(eff.reads)
        .map(|((pre, leaf_eff), reads)| {
            (!pre.is_empty() || !leaf_eff.is_empty() || reads)
                .then_some(LeafPart { pre, eff: leaf_eff })
        })
        .collect();

    GlobalAction {
        op,
        cost,
        center_pre: pre.center,
        center_eff: eff.center,
        parts,
    }
}
