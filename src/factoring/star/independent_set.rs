//! The independent-set strategy of Gnad, Poser and Hoffmann (IJCAI 2017,
//! Section 4.1 and Figure 1), which maximises the number of leaves.
//!
//! Leaves of a strict-star factoring are pairwise non-adjacent in the
//! causal graph, so a maximum independent set of it gives the most leaves
//! possible. Each such set seeds singleton leaves, which are then grown:
//!
//! ```mermaid
//! graph LR
//!     mis[maximum independent sets] --> seed[one leaf per variable]
//!     seed --> grow1[grow: maximise mobile leaves]
//!     grow1 --> frozen[drop frozen leaves]
//!     frozen --> grow2[grow again]
//!     grow2 --> best[the seed with the most mobile leaves]
//! ```
//!
//! Growing (the paper's `maximizeFlexibility`) moves, per round, one center
//! variable adjacent to exactly one leaf into that leaf, so no arc ever
//! joins two leaves; of the round's choices it keeps the one with the most
//! mobile leaves, and of all rounds it returns the factoring with the most
//! mobile leaves. The pseudocode's per-round `argmax(|F_j^L|)` is read as
//! the prose says, by mobile leaves (the number of leaves never changes
//! while growing). Ties are ours: within a round the smallest variable,
//! between rounds the later one, which has the larger leaves (the purpose
//! of the last step). Leaves stay within [`crate::MAX_LEAF_STATES`] states
//! (the paper allows `2^32`; our ranks are `u16`).

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{CausalGraph, OpId, Task, VarId};

use super::{Candidate, mobility};
use crate::decoupled::leaf_action_of;
use crate::factoring::leaf_states;

/// The most maximum independent sets that seed a factoring each.
const MAX_INDEPENDENT_SETS: usize = 16;

/// Neighbour lookups after which the enumeration stops, once it has found
/// at least one set. Replaces the paper's ten-second time-out; past it,
/// the sets found need not be maximum.
const WORK_BUDGET: usize = 1 << 24;

/// The best factoring over all enumerated maximum independent sets: the
/// first with the most mobile leaves.
pub(super) fn factoring(task: &Task) -> Candidate {
    let growth = Growth::new(task);
    let eligible = (0..task.num_variables())
        .filter(|&v| !task.variables[v].is_derived() && leaf_states(task, &[v]).is_some())
        .collect();

    let mut best: Option<Candidate> = None;
    for set in maximum_independent_sets(&growth.neighbours, eligible) {
        let candidate = growth.post_process(set);
        if best.as_ref().is_none_or(|b| candidate.mobile > b.mobile) {
            best = Some(candidate);
        }
    }

    best.unwrap_or(Candidate {
        mobile: 0,
        leaves: Vec::new(),
    })
}

/// What growing leaves needs to know about the task, computed once.
struct Growth<'a> {
    /// The task.
    task: &'a Task,
    /// Per variable, its distinct causal-graph neighbours in either
    /// direction, sorted.
    neighbours: Vec<Vec<VarId>>,
    /// Per variable, the operators that read or write it.
    touching: Vec<Vec<OpId>>,
}

/// Leaves while they grow.
#[derive(Clone)]
struct Leaves {
    /// The variables of each leaf, sorted.
    vars: Vec<Vec<VarId>>,
    /// Per task variable, its leaf; `None` for the center.
    leaf_of: Vec<Option<usize>>,
    /// Per leaf, whether it is mobile.
    mobile: Vec<bool>,
}

impl Leaves {
    /// The number of mobile leaves.
    fn mobile_count(&self) -> usize {
        self.mobile.iter().filter(|&&m| m).count()
    }
}

impl<'a> Growth<'a> {
    /// The neighbours and touching operators of every variable of `task`.
    fn new(task: &'a Task) -> Self {
        let graph = CausalGraph::new(task);
        let neighbours = (0..task.num_variables())
            .map(|v| {
                let mut out: Vec<VarId> = graph
                    .successors(v)
                    .iter()
                    .chain(graph.predecessors(v))
                    .copied()
                    .collect();
                out.sort_unstable();
                out.dedup();
                out
            })
            .collect();

        let mut touching = vec![Vec::new(); task.num_variables()];
        for (op, operator) in task.operators.iter().enumerate() {
            let conditions = operator.eff.iter().flat_map(|e| &e.conditions);
            let effects = operator.eff.iter().map(|e| e.fact.var);
            let mut vars: Vec<VarId> = operator
                .pre
                .iter()
                .chain(conditions)
                .map(|f| f.var)
                .chain(effects)
                .collect();
            vars.sort_unstable();
            vars.dedup();
            for v in vars {
                touching[v].push(op);
            }
        }

        Self {
            task,
            neighbours,
            touching,
        }
    }

    /// `vars` as growing leaves, with their mobility computed from scratch.
    fn leaves(&self, vars: Vec<Vec<VarId>>) -> Leaves {
        let mut leaf_of = vec![None; self.task.num_variables()];
        for (leaf, members) in vars.iter().enumerate() {
            for &v in members {
                leaf_of[v] = Some(leaf);
            }
        }

        let mobile = mobility(self.task, &vars);
        Leaves {
            vars,
            leaf_of,
            mobile,
        }
    }

    /// The paper's post-process of one independent set: grow, drop frozen
    /// leaves, grow again.
    fn post_process(&self, set: Vec<VarId>) -> Candidate {
        let seeded = self.leaves(set.into_iter().map(|v| vec![v]).collect());
        let grown = self.maximize_flexibility(seeded);
        let kept = grown
            .vars
            .into_iter()
            .zip(grown.mobile)
            .filter_map(|(leaf, mobile)| mobile.then_some(leaf))
            .collect();

        let best = self.maximize_flexibility(self.leaves(kept));
        Candidate {
            mobile: best.mobile_count(),
            leaves: best.vars,
        }
    }

    /// Grows `start` one variable per round while any center variable is
    /// adjacent to exactly one leaf, and returns the round with the most
    /// mobile leaves, the latest on ties (`start` itself is round 0).
    fn maximize_flexibility(&self, start: Leaves) -> Leaves {
        let mut best = start.clone();
        let mut current = start;
        while let Some(next) = self.best_extension(&current) {
            current = next;
            if current.mobile_count() >= best.mobile_count() {
                best = current.clone();
            }
        }

        best
    }

    /// Of all ways to move one center variable that is adjacent to exactly
    /// one leaf into it (within the leaf size bound), the one with the most
    /// mobile leaves, the smallest variable on ties; `None` if there is no
    /// such variable.
    fn best_extension(&self, current: &Leaves) -> Option<Leaves> {
        let mut best: Option<(usize, VarId, usize)> = None;
        for v in (0..self.task.num_variables()).filter(|&v| self.may_join(current, v)) {
            let Some(leaf) = only_neighbour_leaf(&self.neighbours[v], &current.leaf_of) else {
                continue;
            };

            if leaf_states(self.task, &joined(&current.vars[leaf], v)).is_none() {
                continue;
            }

            let mobile = self.mobile_count_after(current, v, leaf);
            if best.is_none_or(|(count, _, _)| mobile > count) {
                best = Some((mobile, v, leaf));
            }
        }

        let (_, v, leaf) = best?;
        Some(self.moved(current, v, leaf))
    }

    /// Whether `v` is a center variable that may become a leaf variable.
    fn may_join(&self, current: &Leaves, v: VarId) -> bool {
        current.leaf_of[v].is_none() && !self.task.variables[v].is_derived()
    }

    /// The number of mobile leaves once `v` joins leaf `leaf`.
    ///
    /// Without axioms only `leaf` can change, and only through operators
    /// that touch `v`: an operator touching `v` and writing another leaf
    /// `m` would put an arc between `v` and `m`, but `v` neighbours `leaf`
    /// only. With axioms (derived variables attach to the leaves they read)
    /// everything is recomputed.
    fn mobile_count_after(&self, current: &Leaves, v: VarId, leaf: usize) -> usize {
        if !self.task.axioms.is_empty() {
            let mut vars = current.vars.clone();
            vars[leaf] = joined(&vars[leaf], v);
            return mobility(self.task, &vars).iter().filter(|&&m| m).count();
        }

        let others = current.mobile_count() - usize::from(current.mobile[leaf]);
        others + usize::from(self.becomes_mobile(current, v, leaf))
    }

    /// Whether leaf `leaf` is mobile once `v` (adjacent to no other leaf)
    /// joins it, without axioms.
    fn becomes_mobile(&self, current: &Leaves, v: VarId, leaf: usize) -> bool {
        let leaf_of = |u: VarId| {
            if u == v {
                Some(leaf)
            } else {
                current.leaf_of[u]
            }
        };
        current.mobile[leaf]
            || self.touching[v]
                .iter()
                .any(|&op| leaf_action_of(leaf_of, &self.task.operators[op]) == Some(leaf))
    }

    /// `current` with `v` moved into leaf `leaf`.
    fn moved(&self, current: &Leaves, v: VarId, leaf: usize) -> Leaves {
        let mut vars = current.vars.clone();
        vars[leaf] = joined(&vars[leaf], v);
        if !self.task.axioms.is_empty() {
            return self.leaves(vars);
        }

        let mut next = Leaves {
            vars,
            leaf_of: current.leaf_of.clone(),
            mobile: current.mobile.clone(),
        };
        next.mobile[leaf] = self.becomes_mobile(current, v, leaf);
        next.leaf_of[v] = Some(leaf);
        debug_assert_eq!(
            next.mobile,
            mobility(self.task, &next.vars),
            "incremental mobility"
        );
        next
    }
}

/// `leaf` (sorted) with `v` added, sorted.
fn joined(leaf: &[VarId], v: VarId) -> Vec<VarId> {
    let mut out = leaf.to_vec();
    out.push(v);
    out.sort_unstable();
    out
}

/// The one leaf among `neighbours`' leaves, `None` if they lie in no leaf
/// or in several.
fn only_neighbour_leaf(neighbours: &[VarId], leaf_of: &[Option<usize>]) -> Option<usize> {
    let mut found = None;
    for leaf in neighbours.iter().filter_map(|&u| leaf_of[u]) {
        if *found.get_or_insert(leaf) != leaf {
            return None;
        }
    }

    found
}

/// Up to [`MAX_INDEPENDENT_SETS`] maximum independent sets of the graph
/// induced by `candidates` (sorted), each sorted, in the order found.
///
/// Branch and bound: branch on a vertex of highest degree, first without
/// it, then with it; prune branches that cannot reach the best size (or
/// only tie it once enough sets are found). Within [`WORK_BUDGET`]
/// neighbour lookups the sets are maximum.
fn maximum_independent_sets(neighbours: &[Vec<VarId>], candidates: Vec<VarId>) -> Vec<Vec<VarId>> {
    let mut enumeration = Enumeration {
        neighbours,
        best: 0,
        sets: Vec::new(),
        work: 0,
    };
    enumeration.branch(&mut Vec::new(), candidates);
    enumeration.sets
}

/// The state of [`maximum_independent_sets`].
struct Enumeration<'a> {
    /// Per vertex, its neighbours, sorted.
    neighbours: &'a [Vec<VarId>],
    /// The size of the largest independent set found.
    best: usize,
    /// The independent sets of size [`Enumeration::best`] found, at most
    /// [`MAX_INDEPENDENT_SETS`].
    sets: Vec<Vec<VarId>>,
    /// Neighbour lookups so far, one per search node at least.
    work: usize,
}

impl Enumeration<'_> {
    /// Extends `chosen` by independent subsets of `candidates` (sorted, all
    /// non-adjacent to `chosen`).
    fn branch(&mut self, chosen: &mut Vec<VarId>, candidates: Vec<VarId>) {
        self.work += 1;
        if self.is_hopeless(chosen.len() + candidates.len()) {
            return;
        }

        let Some(v) = self.highest_degree(&candidates) else {
            self.record(chosen, &candidates);
            return;
        };

        let rest: Vec<VarId> = candidates.into_iter().filter(|&u| u != v).collect();
        let apart = rest
            .iter()
            .copied()
            .filter(|u| self.neighbours[v].binary_search(u).is_err())
            .collect();
        self.branch(chosen, rest);

        chosen.push(v);
        self.branch(chosen, apart);
        chosen.pop();
    }

    /// Whether a branch that can reach at most `reachable` vertices cannot
    /// add a set, or the budget is spent.
    fn is_hopeless(&self, reachable: usize) -> bool {
        let spent = self.work > WORK_BUDGET && !self.sets.is_empty();
        let full = self.sets.len() >= MAX_INDEPENDENT_SETS && reachable <= self.best;
        spent || full || reachable < self.best
    }

    /// The candidate with the most neighbours among `candidates`, the
    /// smallest on ties; `None` if they are pairwise non-adjacent.
    fn highest_degree(&mut self, candidates: &[VarId]) -> Option<VarId> {
        self.work += candidates
            .iter()
            .map(|&v| self.neighbours[v].len())
            .sum::<usize>();
        candidates
            .iter()
            .map(|&v| {
                let degree = self.neighbours[v]
                    .iter()
                    .filter(|u| candidates.binary_search(u).is_ok())
                    .count();
                (degree, v)
            })
            .filter(|&(degree, _)| degree > 0)
            .max_by(|(da, va), (db, vb)| da.cmp(db).then(vb.cmp(va)))
            .map(|(_, v)| v)
    }

    /// Records the independent set `chosen ∪ rest`.
    fn record(&mut self, chosen: &[VarId], rest: &[VarId]) {
        let mut set: Vec<VarId> = chosen.iter().chain(rest).copied().collect();
        set.sort_unstable();
        if set.len() > self.best {
            self.best = set.len();
            self.sets.clear();
        }

        if self.sets.len() < MAX_INDEPENDENT_SETS {
            self.sets.push(set);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use sas_plus::random::Rng;

    use super::*;

    /// A random undirected graph over `n` vertices, each edge present with
    /// probability about `percent / 100`, as sorted neighbour lists.
    fn random_graph(rng: &mut Rng, n: usize, percent: u64) -> Vec<Vec<VarId>> {
        let mut neighbours = vec![Vec::new(); n];
        for u in 0..n {
            for v in u + 1..n {
                if rng.range(0, 99) < percent {
                    neighbours[u].push(v);
                    neighbours[v].push(u);
                }
            }
        }

        for list in &mut neighbours {
            list.sort_unstable();
        }

        neighbours
    }

    /// Every independent set of maximum size, by enumerating all subsets.
    fn brute_force(neighbours: &[Vec<VarId>]) -> Vec<Vec<VarId>> {
        let n = neighbours.len();
        let independent = |mask: u32| {
            (0..n).all(|u| mask >> u & 1 == 0 || neighbours[u].iter().all(|&v| mask >> v & 1 == 0))
        };
        let sets: Vec<u32> = (0..1_u32 << n).filter(|&m| independent(m)).collect();
        let best = sets.iter().map(|m| m.count_ones()).max().unwrap_or(0);
        sets.into_iter()
            .filter(|m| m.count_ones() == best)
            .map(|m| (0..n).filter(|&u| m >> u & 1 == 1).collect())
            .collect()
    }

    /// On random graphs of up to eleven vertices, the enumeration returns
    /// distinct maximum independent sets, all of them when there are at
    /// most [`MAX_INDEPENDENT_SETS`].
    #[test]
    fn enumeration_matches_brute_force() {
        let mut rng = Rng::new(7);
        for round in 0..300 {
            let n = usize::try_from(rng.range(0, 11)).expect("small");
            let percent = rng.range(10, 70);
            let neighbours = random_graph(&mut rng, n, percent);
            let expected = brute_force(&neighbours);
            let mut found = maximum_independent_sets(&neighbours, (0..n).collect());

            for set in &found {
                assert!(
                    expected.contains(set),
                    "round {round}: {set:?} is no maximum set"
                );
            }

            found.sort();
            found.dedup();
            let wanted = expected.len().min(MAX_INDEPENDENT_SETS);
            assert_eq!(found.len(), wanted, "round {round}: {neighbours:?}");
        }
    }

    /// Only the candidate vertices take part.
    #[test]
    fn enumeration_respects_candidates() {
        // A star with center 0 and spokes 1, 2, 3; without the spokes as
        // candidates, the center alone is the maximum set.
        let neighbours = vec![vec![1, 2, 3], vec![0], vec![0], vec![0]];
        assert_eq!(
            maximum_independent_sets(&neighbours, vec![0, 1, 2, 3]),
            [[1, 2, 3]]
        );
        assert_eq!(maximum_independent_sets(&neighbours, vec![0]), [[0]]);
    }
}
