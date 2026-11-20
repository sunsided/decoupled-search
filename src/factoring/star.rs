//! Greedy strict-star factorings: move highly connected variables into the
//! center until the rest falls apart into independent leaves.
//!
//! In the spirit of the incident-arcs and maximum-leaves strategies of Gnad,
//! Poser and Hoffmann (IJCAI 2017), but greedy and LP-free; the details are
//! our own and not checked against the paper.
//!
//! ```mermaid
//! graph LR
//!     rest[non-center variables] --> comps[weakly connected components]
//!     comps --> candidate[candidate: small components are leaves]
//!     comps --> pick[move the highest-degree variable to the center]
//!     pick --> rest
//! ```

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{CausalGraph, Task, VarId};

use super::{Factoring, leaf_states};

impl Factoring {
    /// The greedy incident-arcs star factoring: repeatedly moves the variable
    /// with the most causal-graph neighbours outside the center into the
    /// center, and returns the first candidate with at least two leaves.
    ///
    /// A candidate's leaves are the weakly connected components of the
    /// causal graph without the center that have at most
    /// [`crate::MAX_LEAF_STATES`] states; larger components join the center.
    /// No arc connects two leaves, so the result always passes
    /// [`Factoring::check_star`]. If no candidate has two leaves, the result
    /// is [`Factoring::max_leaves`].
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::Factoring;
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A hub (variable 0) that both spokes read and that reads both spokes.
    /// let op = |pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("hub", 2), Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![
    ///         op(vec![Fact::new(1, 1), Fact::new(2, 1)], Fact::new(0, 1)),
    ///         op(vec![Fact::new(0, 0)], Fact::new(1, 1)),
    ///         op(vec![Fact::new(0, 0)], Fact::new(2, 1)),
    ///     ],
    ///     initial: State(vec![0, 0, 0]),
    ///     goal: vec![Fact::new(0, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let factoring = Factoring::incident_arcs(&task);
    /// assert_eq!(factoring.center(), &[0]);
    /// assert_eq!(factoring.leaves(), &[vec![1], vec![2]]);
    /// assert_eq!(factoring.check_star(&task), Ok(()));
    /// ```
    #[must_use]
    pub fn incident_arcs(task: &Task) -> Self {
        let candidates = candidates(task);
        let first = candidates.iter().position(|leaves| leaves.len() >= 2);
        let leaves = match first {
            Some(index) => candidates.into_iter().nth(index),
            None => most_leaves(candidates),
        };

        Self::from_leaves(task, leaves.unwrap_or_default())
    }

    /// The greedy maximum-leaves star factoring: of all candidates
    /// [`Factoring::incident_arcs`] walks through, the one with the most
    /// leaves, preferring the smaller center on ties.
    ///
    /// Always passes [`Factoring::check_star`].
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::Factoring;
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A chain a - b - c: putting b into the center leaves a and c apart.
    /// let op = |pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let var = |name: &str| Variable::new(name, 2);
    /// let task = Task {
    ///     variables: vec![var("a"), var("b"), var("c")],
    ///     operators: vec![
    ///         op(vec![Fact::new(0, 1)], Fact::new(1, 1)),
    ///         op(vec![Fact::new(1, 1)], Fact::new(2, 1)),
    ///         op(vec![], Fact::new(0, 1)),
    ///     ],
    ///     initial: State(vec![0, 0, 0]),
    ///     goal: vec![Fact::new(2, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let factoring = Factoring::max_leaves(&task);
    /// assert_eq!(factoring.center(), &[1]);
    /// assert_eq!(factoring.leaves(), &[vec![0], vec![2]]);
    /// ```
    #[must_use]
    pub fn max_leaves(task: &Task) -> Self {
        Self::from_leaves(task, most_leaves(candidates(task)).unwrap_or_default())
    }
}

/// The candidate with the most leaves, the earliest (smallest center) on
/// ties; `None` only if there are no candidates.
fn most_leaves(candidates: Vec<Vec<Vec<VarId>>>) -> Option<Vec<Vec<VarId>>> {
    candidates.into_iter().rev().max_by_key(Vec::len)
}

/// The leaf sets of the greedy walk, starting with an empty center and
/// growing it by one highest-degree variable per step until every variable
/// is in the center.
///
/// Each candidate's leaves are the small weakly connected components of the
/// non-center variables, each sorted, in order of their smallest variable.
fn candidates(task: &Task) -> Vec<Vec<Vec<VarId>>> {
    let n = task.num_variables();
    let graph = CausalGraph::new(task);
    let mut in_center = vec![false; n];
    let mut candidates = Vec::with_capacity(n + 1);
    loop {
        let leaves = weak_components(&graph, &in_center)
            .into_iter()
            .filter(|c| leaf_states(task, c).is_some())
            .collect();
        candidates.push(leaves);

        match highest_degree(&graph, &in_center) {
            Some(v) => in_center[v] = true,
            None => return candidates,
        }
    }
}

/// The non-center variable with the most distinct causal-graph neighbours
/// outside the center, the smallest on ties; `None` if all are in the center.
fn highest_degree(graph: &CausalGraph, in_center: &[bool]) -> Option<VarId> {
    (0..in_center.len())
        .filter(|&v| !in_center[v])
        .map(|v| (outside_neighbours(graph, in_center, v).len(), v))
        .max_by(|(da, va), (db, vb)| da.cmp(db).then(vb.cmp(va)))
        .map(|(_, v)| v)
}

/// The distinct neighbours of `v` (either arc direction) that are not in
/// the center, sorted.
fn outside_neighbours(graph: &CausalGraph, in_center: &[bool], v: VarId) -> Vec<VarId> {
    let mut neighbours: Vec<VarId> = graph
        .successors(v)
        .iter()
        .chain(graph.predecessors(v))
        .copied()
        .filter(|&u| !in_center[u])
        .collect();
    neighbours.sort_unstable();
    neighbours.dedup();
    neighbours
}

/// The weakly connected components of the causal graph restricted to the
/// non-center variables, each sorted, in order of their smallest variable.
fn weak_components(graph: &CausalGraph, in_center: &[bool]) -> Vec<Vec<VarId>> {
    let mut seen = in_center.to_vec();
    let mut components = Vec::new();
    for start in 0..in_center.len() {
        if seen[start] {
            continue;
        }

        seen[start] = true;
        let mut component = vec![start];
        let mut stack = vec![start];
        while let Some(v) = stack.pop() {
            for u in outside_neighbours(graph, in_center, v) {
                if !seen[u] {
                    seen[u] = true;
                    component.push(u);
                    stack.push(u);
                }
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components
}
