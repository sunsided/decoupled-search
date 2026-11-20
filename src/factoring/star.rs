//! Strict-star factorings of Gnad, Poser and Hoffmann (IJCAI 2017,
//! Section 4): the incident-arcs strategy (their Figure 2) and the
//! independent-set strategy that maximises the number of leaves (their
//! Figure 1, in [`independent_set`]).
//!
//! Both build leaves that no causal-graph arc connects, so every result
//! passes [`Factoring::check_star`]. Both rank their candidates by the
//! number of *mobile* leaves, leaves with at least one leaf action (the
//! paper's flexibility `|A^L[F^L] \ A^C| / |A^L[F^L]|` above 0). The paper
//! abstains below two mobile leaves; these functions return their best
//! candidate anyway, and [`Factoring::auto`] abstains.
//!
//! ```mermaid
//! graph LR
//!     order[variables by incident arcs] --> grow[move the next one to the center]
//!     grow --> comps[small weakly connected components are the leaves]
//!     comps --> count[count mobile leaves]
//!     count --> grow
//!     count --> best[keep the first candidate with the most]
//! ```

mod independent_set;

use alloc::vec;
use alloc::vec::Vec;

use core::cmp::Reverse;

use sas_plus::{CausalGraph, Task, VarId};

use super::{Factoring, leaf_states};
use crate::decoupled::mobile_leaves;

impl Factoring {
    /// The incident-arcs star factoring (Gnad, Poser and Hoffmann 2017,
    /// Figure 2): orders the variables by decreasing number of incident
    /// causal-graph arcs, moves them into the center one at a time, and
    /// keeps the candidate with the most mobile leaves.
    ///
    /// A candidate's leaves are the weakly connected components of the
    /// causal graph without the center that have at most
    /// [`crate::MAX_LEAF_STATES`] states; the other variables form the
    /// center. No arc connects two leaves, so the result always passes
    /// [`Factoring::check_star`]. A variable's incident arcs are its
    /// incoming plus its outgoing arcs, counted once on the full causal
    /// graph. Ties are ours, the paper leaves them open: in the order, the
    /// smaller variable first; between candidates, the smaller center. As
    /// in the paper, the center has at least one variable. The paper
    /// abstains below two mobile leaves; this returns the best
    /// candidate anyway, possibly without leaves.
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
        let graph = CausalGraph::new(task);
        let mut in_center = vec![false; task.num_variables()];
        let mut best: Option<Candidate> = None;
        for v in incident_arc_order(&graph, task.num_variables()) {
            in_center[v] = true;
            let candidate = Candidate::new(task, small_components(task, &graph, &in_center));
            if best.as_ref().is_none_or(|b| candidate.mobile > b.mobile) {
                best = Some(candidate);
            }
        }

        Self::from_leaves(task, best.map(|b| b.leaves).unwrap_or_default())
    }

    /// The independent-set star factoring, which maximises the number of
    /// leaves (Gnad, Poser and Hoffmann 2017, Figure 1): every maximum
    /// independent set of the causal graph seeds one leaf per variable;
    /// then variables adjacent to exactly one leaf join it while that gains
    /// mobile leaves, frozen (immobile) leaves go back to the center, and
    /// leaves grow once more. The result is the candidate with the most
    /// mobile leaves.
    ///
    /// Variables that cannot be leaves (derived, or with more than
    /// [`crate::MAX_LEAF_STATES`] values) are left out of the independent
    /// sets. The paper enumerates maximum independent sets for up to ten
    /// seconds; this enumerates at most 16 of them with a
    /// deterministic search budget, after which the best set found so far
    /// is used. Always passes
    /// [`Factoring::check_star`]. The paper abstains below two mobile
    /// leaves; this returns the best candidate anyway.
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
        Self::from_leaves(task, independent_set::factoring(task).leaves)
    }
}

/// A candidate factoring's leaves and how many of them are mobile.
struct Candidate {
    /// The number of mobile leaves.
    mobile: usize,
    /// The leaves, each sorted.
    leaves: Vec<Vec<VarId>>,
}

impl Candidate {
    /// Scores `leaves`; a factoring the search would reject (an axiom across
    /// two leaves) has no mobile leaves.
    fn new(task: &Task, leaves: Vec<Vec<VarId>>) -> Self {
        let mobile = mobility(task, &leaves).iter().filter(|&&m| m).count();
        Self { mobile, leaves }
    }
}

/// Per leaf, whether it has a leaf action; all `false` if the factoring
/// would be rejected. `leaves` must contain no derived variables, or leaves
/// of only derived variables would shift the indices.
fn mobility(task: &Task, leaves: &[Vec<VarId>]) -> Vec<bool> {
    let factoring = Factoring::from_leaves(task, leaves.to_vec());
    mobile_leaves(task, &factoring).unwrap_or_else(|_| vec![false; factoring.leaves.len()])
}

/// The variables by decreasing number of incident causal-graph arcs
/// (incoming plus outgoing), the smaller variable first on ties.
fn incident_arc_order(graph: &CausalGraph, n: usize) -> Vec<VarId> {
    let mut order: Vec<VarId> = (0..n).collect();
    order.sort_by_key(|&v| {
        (
            Reverse(graph.successors(v).len() + graph.predecessors(v).len()),
            v,
        )
    });
    order
}

/// The weakly connected components of the non-center variables with at
/// most [`crate::MAX_LEAF_STATES`] states.
fn small_components(task: &Task, graph: &CausalGraph, in_center: &[bool]) -> Vec<Vec<VarId>> {
    weak_components(graph, in_center)
        .into_iter()
        .filter(|c| leaf_states(task, c).is_some())
        .collect()
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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use sas_plus::random::{RandomTaskParams, random_task};

    use super::*;

    /// The cheap mobility count the strategies rank by agrees with
    /// [`Factoring::score`] whenever the search accepts the factoring.
    #[test]
    fn mobility_matches_the_score() {
        let mut compared = 0;
        for seed in 0..200 {
            let task = random_task(seed, RandomTaskParams::default());
            let n = task.num_variables();
            let partitions: [Vec<Vec<VarId>>; 3] = [
                (0..n).map(|v| vec![v]).collect(),
                (0..n).step_by(2).map(|v| vec![v]).collect(),
                vec![(0..n / 2).collect()],
            ];
            for leaves in partitions {
                let leaves: Vec<Vec<VarId>> = leaves
                    .into_iter()
                    .filter(|l| !l.is_empty() && leaf_states(&task, l).is_some())
                    .collect();
                let Ok(score) = Factoring::from_leaves(&task, leaves.clone()).score(&task) else {
                    continue;
                };

                let mobile = mobility(&task, &leaves).iter().filter(|&&m| m).count();
                assert_eq!(mobile, score.mobile_leaves, "seed {seed}: {leaves:?}");
                compared += 1;
            }
        }

        assert!(compared > 300, "only {compared} comparisons");
    }
}
