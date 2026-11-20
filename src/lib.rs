//! # Decoupled search
//!
//! Decoupled (star-topology) search for factored SAS+ planning tasks from
//! [`sas_plus`] (Gnad & Hoffmann, ICAPS 2015 and AIJ 2018).
//!
//! A [`Factoring`] splits the variables into a *center* and *leaves*.
//! [`search()`], [`astar`] and [`gbfs`] then explore paths of *global
//! actions* (operators that are not confined to one leaf) and keep, for
//! every leaf, the price of each leaf state reachable along the current
//! path. They return a sequential [`Plan`] for the original task,
//! cost-optimal for the first two; [`Plan::validate`] checks one
//! independently. [`Config`] chooses the dominance criterion ([`Pruning`])
//! and whether leaves keep prices at all ([`Leaves`]).
//!
//! ```mermaid
//! graph LR
//!     task[sas_plus::Task] --> factoring[Factoring: auto, fork, star, explicit]
//!     factoring --> search[search / astar / gbfs over decoupled states]
//!     task --> search
//!     config[Config: pruning, leaves] --> search
//!     heuristic[HMax / HFf / MasHeuristic] --> search
//!     search --> plan[Plan: sequential, validated]
//! ```
//!
//! ## Status
//!
//! Pre-release, see `docs/design.md`: fork, inverted-fork, greedy star and
//! explicit factorings, scored and chosen by [`Factoring::auto`];
//! uniform-cost, A* and greedy search with basic, frontier,
//! effective-price and simulation dominance pruning; reachability-only
//! leaves for greedy search; `h^max` and `h^FF` on decoupled states, and a
//! merge-and-shrink adapter behind the `merge-and-shrink` feature.
//! Conditional effects and axioms are supported where the decoupled
//! semantics can express them and rejected with precise errors otherwise.
//! The API will change.
//!
//! Repository: <https://github.com/sunsided/decoupled-search>

#![no_std]

extern crate alloc;

mod decoupled;
mod factoring;
mod heuristic;
mod plan;
mod search;

use core::fmt;

use sas_plus::{OpId, TaskError, VarId};

pub use factoring::{Factoring, FactoringScore, Strategy};
#[cfg(feature = "merge-and-shrink")]
pub use heuristic::MasHeuristic;
pub use heuristic::{DecoupledHeuristic, FactPrices, HFf, HMax};
pub use plan::{Plan, PlanError};
pub use search::{Config, Leaves, Outcome, Pruning, Statistics, astar, gbfs, search};

/// The largest number of states a leaf may have. Leaf reachability works on
/// dense price vectors of this size, and stored leaf-state ranks fit in
/// `u16`.
pub const MAX_LEAF_STATES: usize = 1 << 16;

/// Why a task or factoring cannot be searched.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The task failed [`sas_plus::Task::validate`].
    InvalidTask(TaskError),
    /// Effect `effect` of operator `op` sets a center variable under a
    /// condition on a leaf variable. The successor's center would then
    /// depend on which leaf state the operator is applied in, which a single
    /// decoupled successor cannot express.
    CenterEffectReadsLeaf {
        /// The offending operator.
        op: OpId,
        /// Index of the offending effect in the operator's effect list.
        effect: usize,
    },
    /// Effect `effect` of operator `op` sets a variable of one leaf under a
    /// condition on another leaf, which would couple the two leaves' states.
    EffectConditionAcrossLeaves {
        /// The offending operator.
        op: OpId,
        /// Index of the offending effect in the operator's effect list.
        effect: usize,
    },
    /// Derived variable `var` depends (directly or through other derived
    /// variables) on the state variables of two leaves, so its value is not
    /// a function of the center and one leaf state.
    AxiomAcrossLeaves {
        /// The offending derived variable.
        var: VarId,
        /// Two of the leaves it depends on, smaller index first.
        leaves: (usize, usize),
    },
    /// A leaf names derived variable `var`. Derived variables are never
    /// leaf variables: they stay in the center of a factoring and are
    /// evaluated with the factor they depend on.
    DerivedInLeaf {
        /// The offending derived variable.
        var: VarId,
    },
    /// Leaf prices without costs ([`Leaves::Reachability`]) were requested
    /// for a cost-optimal search, which needs the prices.
    ReachabilityNotOptimal,
    /// A leaf names variable `var`, which the task does not have.
    UnknownVariable {
        /// The offending variable.
        var: VarId,
    },
    /// Variable `var` appears in two leaves, or twice in one.
    DuplicateVariable {
        /// The offending variable.
        var: VarId,
    },
    /// Leaf `leaf` has no variables.
    EmptyLeaf {
        /// Index of the offending leaf.
        leaf: usize,
    },
    /// Leaf `leaf` has more than [`MAX_LEAF_STATES`] states.
    LeafTooLarge {
        /// Index of the offending leaf.
        leaf: usize,
    },
    /// The factoring does not partition exactly the task's variables, so it
    /// was built for a different task.
    FactoringMismatch,
    /// The causal graph has an arc `from -> to` between two different
    /// leaves, so the factoring is not a strict star; see
    /// [`Factoring::check_star`].
    NotStar {
        /// Source variable of the arc.
        from: VarId,
        /// Target variable of the arc.
        to: VarId,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTask(e) => write!(f, "invalid task: {e}"),
            Self::CenterEffectReadsLeaf { op, effect } => write!(
                f,
                "effect {effect} of operator {op} sets the center under a leaf condition"
            ),
            Self::EffectConditionAcrossLeaves { op, effect } => write!(
                f,
                "effect {effect} of operator {op} sets one leaf under a condition on another"
            ),
            Self::AxiomAcrossLeaves {
                var,
                leaves: (a, b),
            } => write!(f, "derived variable {var} depends on leaves {a} and {b}"),
            Self::DerivedInLeaf { var } => {
                write!(f, "derived variable {var} cannot be a leaf variable")
            }
            Self::ReachabilityNotOptimal => {
                write!(f, "reachability-only leaves need a greedy search")
            }
            Self::UnknownVariable { var } => write!(f, "unknown variable {var} in a leaf"),
            Self::DuplicateVariable { var } => write!(f, "variable {var} appears twice"),
            Self::EmptyLeaf { leaf } => write!(f, "leaf {leaf} is empty"),
            Self::LeafTooLarge { leaf } => {
                write!(f, "leaf {leaf} has more than {MAX_LEAF_STATES} states")
            }
            Self::FactoringMismatch => write!(f, "the factoring belongs to a different task"),
            Self::NotStar { from, to } => {
                write!(f, "causal-graph arc {from} -> {to} connects two leaves")
            }
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidTask(e) => Some(e),
            _ => None,
        }
    }
}
