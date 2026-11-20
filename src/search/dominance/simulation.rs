//! Leaf simulation relations (Torralba, Gnad, Dubbert and Hoffmann 2016,
//! Definition 5), generalised to global actions that read and write leaves.
//!
//! `x ⪯ y` ("`y` simulates `x`") means that whatever the leaf can still do
//! from `x`, it can do from `y` at no higher cost, under every future center
//! path. The relation is the greatest fixpoint of these conditions on pairs
//! of leaf states:
//!
//! 1. if `x` satisfies the leaf goal, so does `y`;
//! 2. for every global action with a part `P` on the leaf: if `x` satisfies
//!    `P`'s leaf preconditions, so does `y`, and `P(x) ⪯ P(y)`. Global
//!    actions are synchronised with the center, so they must be matched by
//!    the same action;
//! 3. for every leaf action `a` applicable in `x` with `x -a-> x'`: either
//!    `x' ⪯ y` (the leaf stays at `y`), or some leaf action `b` with
//!    `cost(b) <= cost(a)` and center preconditions contained in `a`'s is
//!    applicable in `y` with `y -b-> y'` and `x' ⪯ y'`.
//!
//! *Soundness.* Let `π` be any leaf path from `x` that complies with a center
//! path: leaf actions whose center preconditions hold where they are taken,
//! and leaf parts of the center path's global actions. Induction on `π`
//! with the conditions above yields a leaf path from `y` complying with the
//! same center path (each `b` is applicable wherever `a` was, and each
//! global action's leaf preconditions hold), at no higher cost, ending in a
//! goal leaf state whenever `π` does. The relation is reflexive and
//! transitive (composing two such simulations is one).
//!
//! It is only computed for leaves whose transitions do not depend on the
//! center (see [`crate::decoupled::Leaf::reads_center`]): with
//! center-conditioned effects or derived slots, a transition would have to
//! be matched under every center separately.

use alloc::vec;
use alloc::vec::Vec;

use crate::decoupled::{Leaf, LeafPart, View};

/// A set of leaf states as a bit vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bits(Vec<u64>);

impl Bits {
    /// The set of all `n` leaf states, if `full`, or the empty set.
    fn new(n: usize, full: bool) -> Self {
        let mut words = vec![if full { u64::MAX } else { 0 }; n.div_ceil(64)];
        if full && n % 64 != 0 {
            *words.last_mut().expect("n > 0") = (1 << (n % 64)) - 1;
        }

        Self(words)
    }

    /// Whether `i` is in the set.
    pub(crate) fn contains(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 == 1
    }

    /// Removes `i`.
    fn remove(&mut self, i: usize) {
        self.0[i / 64] &= !(1 << (i % 64));
    }

    /// Adds `i`.
    fn insert(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }
}

/// The transitions of a center-independent leaf, per leaf state.
struct Transitions {
    /// Per leaf state: `(leaf action index, successor)` for every applicable
    /// leaf action.
    leaf: Vec<Vec<(usize, usize)>>,
    /// Per leaf state and global part: the successor, `None` if the state
    /// does not satisfy the part's leaf preconditions.
    global: Vec<Vec<Option<usize>>>,
    /// The number of leaf actions.
    actions: usize,
}

/// The coarsest leaf simulation of `leaf`, as the set of simulators of
/// every leaf state: `result[x].contains(y)` iff `x ⪯ y`. `parts` are the
/// leaf parts of all global actions touching the leaf.
///
/// Refines the pairs that pass the static conditions with a worklist: when
/// `(x', y')` leaves the relation, only pairs `(x, y)` with a transition
/// `x -> x'` and `y = y'` or `y -> y'` can change, so only those are
/// rechecked. Each removal costs `O(in-degree^2)` rechecks of
/// `O(actions^2)` each; callers bound the leaf size.
pub(crate) fn simulators(leaf: &Leaf, parts: &[&LeafPart]) -> Vec<Bits> {
    let view = leaf.static_view();
    let transitions = transitions(&view, parts);
    let weaker = weaker_actions(leaf);
    let mut rel = initial_relation(&view, &transitions);
    let predecessors = predecessors(&transitions, leaf.size);

    let mut work: Vec<(usize, usize)> = (0..leaf.size)
        .flat_map(|x| (0..leaf.size).map(move |y| (x, y)))
        .filter(|&(x, y)| x != y && rel[x].contains(y))
        .collect();
    while let Some((x, y)) = work.pop() {
        if !rel[x].contains(y) || simulates(&rel, &transitions, &weaker, x, y) {
            continue;
        }

        rel[x].remove(y);
        for &px in &predecessors[x] {
            let ys = core::iter::once(y).chain(predecessors[y].iter().copied());
            work.extend(
                ys.filter(|&py| px != py && rel[px].contains(py))
                    .map(|py| (px, py)),
            );
        }
    }

    rel
}

/// Per leaf state, the distinct leaf states with a leaf-action or global
/// transition into it.
fn predecessors(transitions: &Transitions, size: usize) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); size];
    for x in 0..size {
        let leaf = transitions.leaf[x].iter().map(|&(_, next)| next);
        for next in leaf.chain(transitions.global[x].iter().flatten().copied()) {
            predecessors[next].push(x);
        }
    }

    for p in &mut predecessors {
        p.sort_unstable();
        p.dedup();
    }

    predecessors
}

/// The leaf-action and global-part transitions of every leaf state.
fn transitions(view: &View<'_>, parts: &[&LeafPart]) -> Transitions {
    let leaf = view.leaf;
    let apply = |rank: usize, part: &LeafPart| {
        view.satisfies(rank, &part.pre)
            .then(|| view.apply(rank, &part.eff))
    };

    Transitions {
        leaf: (0..leaf.size)
            .map(|rank| {
                leaf.actions
                    .iter()
                    .enumerate()
                    .filter_map(|(i, a)| apply(rank, &a.part).map(|next| (i, next)))
                    .collect()
            })
            .collect(),
        global: (0..leaf.size)
            .map(|rank| parts.iter().map(|part| apply(rank, part)).collect())
            .collect(),
        actions: leaf.actions.len(),
    }
}

/// For every pair of leaf actions `(a, b)`, whether `b` may stand in for
/// `a`: no more expensive, and with center preconditions contained in
/// `a`'s. Indexed `a * actions + b`.
fn weaker_actions(leaf: &Leaf) -> Vec<bool> {
    let actions = &leaf.actions;
    actions
        .iter()
        .flat_map(|a| {
            actions
                .iter()
                .map(|b| b.cost <= a.cost && b.center_pre.iter().all(|f| a.center_pre.contains(f)))
        })
        .collect()
}

/// The pairs that pass conditions 1 and the precondition half of 2.
fn initial_relation(view: &View<'_>, transitions: &Transitions) -> Vec<Bits> {
    let n = view.leaf.size;
    (0..n)
        .map(|x| {
            let mut simulators = Bits::new(n, false);
            for y in 0..n {
                let goal = !view.is_goal(x) || view.is_goal(y);
                let pre = transitions.global[x]
                    .iter()
                    .zip(&transitions.global[y])
                    .all(|(gx, gy)| gx.is_none() || gy.is_some());
                if goal && pre {
                    simulators.insert(y);
                }
            }

            simulators
        })
        .collect()
}

/// Whether `y` still simulates `x` under `rel`: conditions 2 and 3.
fn simulates(rel: &[Bits], transitions: &Transitions, weaker: &[bool], x: usize, y: usize) -> bool {
    let global = transitions.global[x]
        .iter()
        .zip(&transitions.global[y])
        .all(|(gx, gy)| match (gx, gy) {
            (Some(nx), Some(ny)) => rel[*nx].contains(*ny),
            (Some(_), None) => false,
            (None, _) => true,
        });

    let count = transitions.actions;
    global
        && transitions.leaf[x].iter().all(|&(a, nx)| {
            rel[nx].contains(y)
                || transitions.leaf[y]
                    .iter()
                    .any(|&(b, ny)| weaker[a * count + b] && rel[nx].contains(ny))
        })
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

    use super::*;
    use crate::Factoring;
    use crate::decoupled::DecoupledTask;

    /// A truck over three locations and a package (values 0..3 at a
    /// location, 3 in the truck) that must reach location 1.
    fn logistics() -> DecoupledTask {
        let op = |pre: Vec<Fact>, eff: Fact| Operator {
            name: "op".into(),
            pre,
            eff: vec![Effect::new(eff)],
            cost: 1,
        };
        let mut operators = Vec::new();
        for x in 0..3 {
            for y in (0..3).filter(|&y| y != x) {
                operators.push(op(vec![Fact::new(0, x)], Fact::new(0, y)));
            }

            operators.push(op(vec![Fact::new(0, x), Fact::new(1, x)], Fact::new(1, 3)));
            operators.push(op(vec![Fact::new(0, x), Fact::new(1, 3)], Fact::new(1, x)));
        }

        let task = Task {
            variables: vec![Variable::new("truck", 3), Variable::new("pkg", 4)],
            operators,
            initial: State(vec![0, 0]),
            goal: vec![Fact::new(1, 1)],
            axioms: vec![],
        };
        DecoupledTask::new(&task, &Factoring::fork(&task)).expect("supported")
    }

    /// In the truck simulates every non-goal location, since all a package
    /// can do there is to be loaded. At the goal simulates everything (all
    /// a package can still do is to get there), but is simulated only by
    /// itself.
    #[test]
    fn truck_simulates_the_other_locations() {
        let task = logistics();
        let rel = simulators(&task.leaves[0], &[]);
        let simulators_of =
            |x: usize| -> Vec<usize> { (0..4).filter(|&y| rel[x].contains(y)).collect() };

        assert_eq!(simulators_of(0), [0, 1, 3]);
        assert_eq!(simulators_of(2), [1, 2, 3]);
        assert_eq!(simulators_of(1), [1]);
        assert_eq!(simulators_of(3), [1, 3]);
    }
}
