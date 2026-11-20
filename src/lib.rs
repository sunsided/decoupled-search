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
//! cost-optimal for the first two.
//!
//! ```mermaid
//! graph LR
//!     task[sas_plus::Task] --> factoring[Factoring: fork, star, explicit]
//!     factoring --> search[search / astar / gbfs over decoupled states]
//!     task --> search
//!     heuristic[HMax / HFf] --> search
//!     search --> plan[Plan: sequential]
//! ```
//!
//! ## Status
//!
//! Pre-release, see `docs/design.md`: fork, inverted-fork, greedy star and
//! explicit factorings; uniform-cost, A* and greedy search with leaf-state
//! dominance pruning; `h^max` and `h^FF` on decoupled states. Tasks with
//! axioms or conditional effects are rejected. The API will change.
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

pub use factoring::Factoring;
pub use heuristic::{DecoupledHeuristic, FactPrices, HFf, HMax};
pub use plan::{Plan, PlanError};
pub use search::{astar, gbfs, search};

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
    /// The task has axioms or derived variables, which are not supported yet.
    Axioms,
    /// Operator `op` has a conditional effect, which is not supported yet.
    ConditionalEffects {
        /// The offending operator.
        op: OpId,
    },
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
            Self::Axioms => write!(f, "axioms and derived variables are not supported"),
            Self::ConditionalEffects { op } => {
                write!(
                    f,
                    "operator {op} has conditional effects, which are not supported"
                )
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
