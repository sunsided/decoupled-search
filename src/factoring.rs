//! Center/leaf partitions of a task's variables.

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{CausalGraph, Task, VarId};

use crate::{Error, MAX_LEAF_STATES};

/// A partition of a task's variables into one center and any number of
/// leaves.
///
/// Every partition is valid: operators that would break the star topology
/// (for example by reading one leaf and writing another) are searched
/// explicitly as global actions. The factoring only decides how much of the
/// task is priced in the leaves instead of searched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Factoring {
    /// Center variables, sorted ascending.
    center: Vec<VarId>,
    /// Leaf variables, each leaf non-empty and with at most
    /// [`MAX_LEAF_STATES`] states. No variable appears twice.
    leaves: Vec<Vec<VarId>>,
}

impl Factoring {
    /// A factoring with the given leaves; all other variables form the center.
    ///
    /// # Errors
    ///
    /// - [`Error::UnknownVariable`] if a leaf names a variable `task` lacks.
    /// - [`Error::DuplicateVariable`] if a variable appears twice.
    /// - [`Error::EmptyLeaf`] if a leaf has no variables.
    /// - [`Error::LeafTooLarge`] if a leaf has more than [`MAX_LEAF_STATES`]
    ///   states.
    pub fn new(task: &Task, leaves: Vec<Vec<VarId>>) -> Result<Self, Error> {
        let mut in_leaf = vec![false; task.num_variables()];
        for (index, leaf) in leaves.iter().enumerate() {
            if leaf.is_empty() {
                return Err(Error::EmptyLeaf { leaf: index });
            }

            for &var in leaf {
                let seen = in_leaf.get_mut(var).ok_or(Error::UnknownVariable { var })?;
                if *seen {
                    return Err(Error::DuplicateVariable { var });
                }

                *seen = true;
            }

            if leaf_states(task, leaf).is_none() {
                return Err(Error::LeafTooLarge { leaf: index });
            }
        }

        Ok(Self::from_leaves(task, leaves))
    }

    /// A factoring with the given, already checked leaves; all other
    /// variables form the center.
    fn from_leaves(task: &Task, leaves: Vec<Vec<VarId>>) -> Self {
        let mut in_leaf = vec![false; task.num_variables()];
        for &v in leaves.iter().flatten() {
            in_leaf[v] = true;
        }

        let center = (0..task.num_variables()).filter(|&v| !in_leaf[v]).collect();
        Self { center, leaves }
    }

    /// The fork factoring: every sink strongly connected component of the
    /// causal graph (one no arc leaves) with at most [`MAX_LEAF_STATES`]
    /// states is a leaf, everything else is the center.
    ///
    /// Operators on a sink component read only the center and that component,
    /// so all of them are priced in the leaf. A task whose causal graph is one
    /// small component becomes one leaf with an empty center.
    #[must_use]
    pub fn fork(task: &Task) -> Self {
        let leaves: Vec<Vec<VarId>> =
            sink_components(&CausalGraph::new(task), task.num_variables())
                .into_iter()
                .filter(|c| leaf_states(task, c).is_some())
                .collect();

        // Sink components are disjoint, non-empty and filtered by size: nothing to check.
        Self::from_leaves(task, leaves)
    }

    /// The center variables, sorted ascending. May be empty.
    #[must_use]
    pub fn center(&self) -> &[VarId] {
        &self.center
    }

    /// The leaves, each a non-empty list of variables.
    #[must_use]
    pub fn leaves(&self) -> &[Vec<VarId>] {
        &self.leaves
    }
}

/// The number of states of a leaf over `vars`, or `None` if it exceeds
/// [`MAX_LEAF_STATES`].
fn leaf_states(task: &Task, vars: &[VarId]) -> Option<usize> {
    vars.iter().try_fold(1_usize, |size, &v| {
        let domain = usize::try_from(task.variables[v].domain_size).ok()?;
        size.checked_mul(domain).filter(|&s| s <= MAX_LEAF_STATES)
    })
}

/// The sink strongly connected components of `graph` over `n` variables,
/// each sorted, in order of their smallest variable.
///
/// A vertex's component is a sink iff every vertex it reaches reaches it back.
///
/// shortcut: one depth-first search per vertex, `O(n * (n + arcs))` time and
/// `O(n^2)` memory; switch to Tarjan if tasks with thousands of variables
/// show up.
fn sink_components(graph: &CausalGraph, n: usize) -> Vec<Vec<VarId>> {
    let reach: Vec<Vec<bool>> = (0..n).map(|v| reachable(graph, n, v)).collect();
    let mut assigned = vec![false; n];
    let mut sinks = Vec::new();
    for v in 0..n {
        if assigned[v] {
            continue;
        }

        let reached: Vec<VarId> = (0..n).filter(|&u| reach[v][u]).collect();
        for &u in &reached {
            assigned[u] |= reach[u][v];
        }

        if reached.iter().all(|&u| reach[u][v]) {
            sinks.push(reached);
        }
    }

    sinks
}

/// The vertices reachable from `start` in `graph`, including `start`.
fn reachable(graph: &CausalGraph, n: usize, start: VarId) -> Vec<bool> {
    let mut seen = vec![false; n];
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(v) = stack.pop() {
        for &w in graph.successors(v) {
            if !seen[w] {
                seen[w] = true;
                stack.push(w);
            }
        }
    }

    seen
}
