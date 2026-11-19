//! # Decoupled search
//!
//! Decoupled search for factored SAS+ planning: center and leaf factoring, leaf reachability and leaf-state dominance.
//!
//! ## Planned scope
//!
//! - Center/leaf factorings of SAS+ tasks
//! - Decoupled state representation and leaf reachability
//! - Leaf-state dominance pruning
//! - Optional heuristics from merge-and-shrink
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/decoupled-search>

#![no_std]

/// A partition of a task's variables into one center and several leaves.
pub trait Factoring {
    /// The variables in the center factor.
    fn center_variables(&self) -> &[usize];

    /// The number of leaf factors.
    fn leaf_count(&self) -> usize;

    /// The variables in leaf factor `leaf`.
    fn leaf_variables(&self, leaf: usize) -> &[usize];
}
