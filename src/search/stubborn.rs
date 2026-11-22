//! Decoupled strong stubborn sets (Gnad, Hoffmann and Wehrle, JAIR 2019,
//! Definitions 4 to 8): which global actions of an expanded decoupled state
//! need to be applied. [`crate::Config::stubborn_sets`] documents the rules
//! and where their guarantee stops; `docs/design.md` maps them to the paper.

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Fact, OpId, Task, Value};

use crate::decoupled::{DecoupledState, DecoupledTask, LeafFact, Price, UNREACHED, View};

/// A partial assignment split by factor, for the "reached" tests.
#[derive(Clone, Debug, Default)]
struct Condition {
    /// The facts in variable order, as in the task.
    facts: Vec<Fact>,
    /// The facts on center variables, as `(center slot, task fact)`.
    center: Vec<(usize, Fact)>,
    /// The facts on each leaf the condition mentions, in leaf order:
    /// the leaf, its facts by slot, and the same facts as task facts.
    leaves: Vec<(usize, Vec<LeafFact>, Vec<Fact>)>,
}

/// Where a task variable lives.
#[derive(Clone, Copy, Debug)]
enum Place {
    /// Center slot.
    Center(usize),
    /// Leaf index and slot within the leaf.
    Leaf(usize, usize),
}

/// The center-ignoring transition graph of one leaf: every leaf action and
/// every global part on the leaf, applicable wherever its leaf
/// preconditions hold, with global parts at cost 0 (their cost is in `g`).
/// Transitions removed by leaf reduction are kept, so the sets built on it
/// are supersets of those on the reduced graph.
#[derive(Clone, Debug)]
struct LeafGraph {
    /// Outgoing transitions of each leaf state: `(operator, target, cost)`.
    out: Vec<Vec<(OpId, usize, Price)>>,
    /// Incoming transitions of each leaf state: `(source, cost)`.
    into: Vec<Vec<(usize, Price)>>,
    /// Cheapest cost from the initial leaf state.
    from_initial: Vec<Price>,
    /// Cheapest cost to a leaf state satisfying the leaf goal.
    to_goal: Vec<Price>,
}

/// The precomputed parts of the stubborn set construction for one task.
#[derive(Clone, Debug)]
pub(crate) struct StubbornSets {
    /// Precondition of every operator.
    pre: Vec<Condition>,
    /// Effects of every operator in variable order (last one per variable).
    eff: Vec<Vec<Fact>>,
    /// The goal.
    goal: Condition,
    /// The operators that achieve each fact, by variable and value.
    achievers: Vec<Vec<Vec<OpId>>>,
    /// Global action index of each operator; `None` for leaf actions.
    global_of: Vec<Option<usize>>,
    /// The center-ignoring graph of each leaf.
    graphs: Vec<LeafGraph>,
    /// Per operator, the operators that interfere with it and whose
    /// precondition is consistent with its own; filled on first use.
    interference: Vec<Option<Vec<OpId>>>,
    /// Per `(operator, leaf)`, the leaf states from which the operator's
    /// precondition on the leaf can be reached; filled on first use.
    can_reach: BTreeMap<(OpId, usize), Vec<bool>>,
}

impl StubbornSets {
    /// The construction for `task` compiled as `decoupled`; `None` if the
    /// task has conditional effects or axioms, which the paper's
    /// construction does not cover.
    pub(crate) fn new(task: &Task, decoupled: &DecoupledTask) -> Option<Self> {
        let conditional = task
            .operators
            .iter()
            .any(|op| op.eff.iter().any(|e| !e.conditions.is_empty()));
        if conditional || !task.axioms.is_empty() {
            return None;
        }

        let mut place = vec![Place::Center(0); task.num_variables()];
        for (slot, &var) in decoupled.center_vars.iter().enumerate() {
            place[var] = Place::Center(slot);
        }
        for (leaf, l) in decoupled.leaves.iter().enumerate() {
            for (slot, &var) in l.vars.iter().enumerate() {
                place[var] = Place::Leaf(leaf, slot);
            }
        }

        let condition = |facts: &[Fact]| condition(&place, facts);
        let mut achievers: Vec<Vec<Vec<OpId>>> = task
            .variables
            .iter()
            .map(|v| vec![Vec::new(); widen(v.domain_size)])
            .collect();
        let mut eff = Vec::with_capacity(task.operators.len());
        for (id, op) in task.operators.iter().enumerate() {
            let mut facts: Vec<Fact> = Vec::new();
            for e in &op.eff {
                facts.retain(|f| f.var != e.fact.var);
                facts.push(e.fact);
            }
            facts.sort_unstable();
            for f in &facts {
                achievers[f.var][widen(f.value)].push(id);
            }
            eff.push(facts);
        }

        let mut global_of = vec![None; task.operators.len()];
        for (index, global) in decoupled.globals.iter().enumerate() {
            global_of[global.op] = Some(index);
        }

        let graphs = (0..decoupled.leaves.len())
            .map(|leaf| LeafGraph::new(decoupled, leaf))
            .collect();

        Some(Self {
            pre: task.operators.iter().map(|op| condition(&op.pre)).collect(),
            eff,
            goal: condition(&task.goal),
            achievers,
            global_of,
            graphs,
            interference: vec![None; task.operators.len()],
            can_reach: BTreeMap::new(),
        })
    }

    /// Which global actions to apply in `state`, by global action index,
    /// and how many reached global actions that leaves out.
    pub(crate) fn allowed(
        &mut self,
        task: &DecoupledTask,
        state: &DecoupledState,
    ) -> (Vec<bool>, u64) {
        let dense: Vec<Vec<Price>> = state
            .prices
            .iter()
            .zip(&task.leaves)
            .map(|(prices, leaf)| prices.to_dense(leaf.size))
            .collect();
        let views: Vec<View<'_>> = task.leaves.iter().map(|l| l.static_view()).collect();
        let reached = Reached {
            center: &state.center,
            dense: &dense,
            views: &views,
        };

        let mut set = vec![false; self.pre.len()];
        let mut queue = Vec::new();
        let add = |ops: &[OpId], set: &mut Vec<bool>, queue: &mut Vec<OpId>| {
            for &op in ops {
                if !set[op] {
                    set[op] = true;
                    queue.push(op);
                }
            }
        };

        if task.goal_price(state).is_some() {
            let frontier = self.goal_frontier(&reached, task);
            add(&frontier, &mut set, &mut queue);
        } else {
            let enabling = self
                .necessary_enabling(&self.goal, &reached)
                .expect("a non-goal state misses part of the goal");
            add(&enabling, &mut set, &mut queue);
        }

        while let Some(op) = queue.pop() {
            if let Some(enabling) = self.necessary_enabling(&self.pre[op], &reached) {
                add(&enabling, &mut set, &mut queue);
                continue;
            }

            let interfering = self.interfering(op);
            add(&interfering, &mut set, &mut queue);
            for index in 0..self.pre[op].leaves.len() {
                let enabling = self.reached_enabling(op, index, &reached);
                add(&enabling, &mut set, &mut queue);
            }
        }

        let mut allowed = vec![false; task.globals.len()];
        let mut pruned = 0;
        for (op, &global) in self.global_of.iter().enumerate() {
            let Some(global) = global else { continue };
            if set[op] {
                allowed[global] = true;
            } else if reached.condition(&self.pre[op]) {
                pruned += 1;
            }
        }

        (allowed, pruned)
    }

    /// A decoupled necessary enabling set for `condition` (Definition 4):
    /// the achievers of the first unreached leaf fact; else of every fact
    /// of the first leaf whose part is not reached by one leaf state; else
    /// of the first center fact that does not hold. `None` if `condition`
    /// is reached.
    fn necessary_enabling(
        &self,
        condition: &Condition,
        reached: &Reached<'_>,
    ) -> Option<Vec<OpId>> {
        let achievers = |f: &Fact| self.achievers[f.var][widen(f.value)].iter().copied();
        for (leaf, facts, task_facts) in &condition.leaves {
            let unreached = facts
                .iter()
                .position(|f| !reached.leaf(*leaf, core::slice::from_ref(f)));
            if let Some(i) = unreached {
                return Some(achievers(&task_facts[i]).collect());
            }
        }

        for (leaf, facts, task_facts) in &condition.leaves {
            if !reached.leaf(*leaf, facts) {
                return Some(task_facts.iter().flat_map(achievers).collect());
            }
        }

        let (_, missing) = condition
            .center
            .iter()
            .find(|&&(slot, f)| reached.center[slot] != f.value)?;
        Some(achievers(missing).collect())
    }

    /// The operators that interfere with `op` and whose precondition is
    /// consistent with `op`'s (rule 3).
    fn interfering(&mut self, op: OpId) -> Vec<OpId> {
        if let Some(ops) = &self.interference[op] {
            return ops.clone();
        }

        let ops: Vec<OpId> = (0..self.pre.len())
            .filter(|&other| {
                other != op
                    && !conflict(&self.pre[op].facts, &self.pre[other].facts)
                    && (conflict(&self.eff[op], &self.pre[other].facts)
                        || conflict(&self.eff[other], &self.pre[op].facts)
                        || conflict(&self.eff[op], &self.eff[other]))
            })
            .collect();
        self.interference[op] = Some(ops.clone());
        ops
    }

    /// A reached-enabling set for the `index`-th leaf part of `op`'s
    /// precondition (Definition 5): every operator on a transition out of
    /// a reached leaf state that violates the part, into a leaf state from
    /// which the part can be reached.
    fn reached_enabling(&mut self, op: OpId, index: usize, reached: &Reached<'_>) -> Vec<OpId> {
        let (leaf, ref facts, _) = self.pre[op].leaves[index];
        let view = &reached.views[leaf];
        let graph = &self.graphs[leaf];
        let can_reach = self.can_reach.entry((op, leaf)).or_insert_with(|| {
            let mut seen: Vec<bool> = (0..graph.out.len())
                .map(|x| view.satisfies(x, facts))
                .collect();
            let mut stack: Vec<usize> = (0..seen.len()).filter(|&x| seen[x]).collect();
            while let Some(y) = stack.pop() {
                for &(x, _) in &graph.into[y] {
                    if !seen[x] {
                        seen[x] = true;
                        stack.push(x);
                    }
                }
            }
            seen
        });

        let mut ops = Vec::new();
        for (x, &price) in reached.dense[leaf].iter().enumerate() {
            if price == UNREACHED || !can_reach[x] || view.satisfies(x, facts) {
                continue;
            }

            ops.extend(
                graph.out[x]
                    .iter()
                    .filter(|&&(_, y, _)| can_reach[y])
                    .map(|&(a, _, _)| a),
            );
        }

        ops
    }

    /// The goal-price frontier sets of every leaf with a goal (Definition
    /// 7): the operators on transitions `x -a-> y` with `x` reached,
    /// `price(x) + cost(a) < price(y)`, and `g*(x) + cost(a) + h*(y)` below
    /// the leaf's goal price.
    fn goal_frontier(&self, reached: &Reached<'_>, task: &DecoupledTask) -> Vec<OpId> {
        let mut ops = Vec::new();
        for (leaf, l) in task.leaves.iter().enumerate() {
            if l.goal.is_empty() {
                continue;
            }

            let prices = reached.dense[leaf].as_slice();
            let view = &reached.views[leaf];
            let goal_price = (0..prices.len())
                .filter(|&x| view.is_goal(x))
                .map(|x| prices[x])
                .min()
                .unwrap_or(UNREACHED);
            let graph = &self.graphs[leaf];
            for (x, &price) in prices.iter().enumerate() {
                if price == UNREACHED {
                    continue;
                }

                for &(a, y, cost) in &graph.out[x] {
                    let improves = price.saturating_add(cost) < prices[y];
                    let useful = graph.from_initial[x]
                        .saturating_add(cost)
                        .saturating_add(graph.to_goal[y])
                        < goal_price;
                    if improves && useful {
                        ops.push(a);
                    }
                }
            }
        }

        ops
    }
}

/// The reached tests on one decoupled state.
struct Reached<'a> {
    /// The center state.
    center: &'a [Value],
    /// Dense prices of every leaf.
    dense: &'a [Vec<Price>],
    /// Center-ignoring views of every leaf.
    views: &'a [View<'a>],
}

impl Reached<'_> {
    /// Whether some reached leaf state of `leaf` satisfies `facts`.
    fn leaf(&self, leaf: usize, facts: &[LeafFact]) -> bool {
        let view = &self.views[leaf];
        self.dense[leaf]
            .iter()
            .enumerate()
            .any(|(x, &p)| p != UNREACHED && view.satisfies(x, facts))
    }

    /// Whether `condition` is reached: its center part holds and each of
    /// its leaf parts holds in one reached leaf state.
    fn condition(&self, condition: &Condition) -> bool {
        condition
            .center
            .iter()
            .all(|&(slot, f)| self.center[slot] == f.value)
            && condition
                .leaves
                .iter()
                .all(|(leaf, facts, _)| self.leaf(*leaf, facts))
    }
}

impl LeafGraph {
    /// The graph of leaf `leaf` of `task`.
    fn new(task: &DecoupledTask, leaf: usize) -> Self {
        let l = &task.leaves[leaf];
        let view = l.static_view();
        let mut out = vec![Vec::new(); l.size];
        let mut into = vec![Vec::new(); l.size];
        for (x, moves) in out.iter_mut().enumerate() {
            let leaf_moves = l.actions.iter().map(|a| (a.op, &a.part, a.cost));
            let global_moves = task
                .globals
                .iter()
                .filter_map(|g| g.parts[leaf].as_ref().map(|part| (g.op, part, 0)));
            for (op, part, cost) in leaf_moves.chain(global_moves) {
                if view.satisfies(x, &part.pre) {
                    let y = view.apply(x, &part.eff);
                    moves.push((op, y, cost));
                    into[y].push((x, cost));
                }
            }
        }

        let from_initial = dijkstra(
            &[l.initial],
            |x| out[x].iter().map(|&(_, y, c)| (y, c)),
            l.size,
        );
        let goals: Vec<usize> = (0..l.size).filter(|&x| view.is_goal(x)).collect();
        let to_goal = dijkstra(&goals, |y| into[y].iter().copied(), l.size);
        Self {
            out,
            into,
            from_initial,
            to_goal,
        }
    }
}

/// Cheapest costs from `sources` over `edges`, [`UNREACHED`] where none.
fn dijkstra<I: Iterator<Item = (usize, Price)>>(
    sources: &[usize],
    edges: impl Fn(usize) -> I,
    size: usize,
) -> Vec<Price> {
    let mut distance = vec![UNREACHED; size];
    let mut open = BinaryHeap::new();
    for &s in sources {
        distance[s] = 0;
        open.push(Reverse((0, s)));
    }

    while let Some(Reverse((d, x))) = open.pop() {
        if d > distance[x] {
            continue;
        }

        for (y, cost) in edges(x) {
            let through = d.saturating_add(cost);
            if through < distance[y] {
                distance[y] = through;
                open.push(Reverse((through, y)));
            }
        }
    }

    distance
}

/// `facts` split by factor.
fn condition(place: &[Place], facts: &[Fact]) -> Condition {
    let mut sorted = facts.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut condition = Condition {
        facts: sorted.clone(),
        ..Condition::default()
    };
    for f in &sorted {
        match place[f.var] {
            Place::Center(slot) => condition.center.push((slot, *f)),
            Place::Leaf(leaf, slot) => {
                let fact = LeafFact {
                    slot,
                    value: widen(f.value),
                };
                match condition.leaves.iter_mut().find(|(l, _, _)| *l == leaf) {
                    Some((_, part, task_facts)) => {
                        part.push(fact);
                        task_facts.push(*f);
                    }
                    None => condition.leaves.push((leaf, vec![fact], vec![*f])),
                }
            }
        }
    }
    condition.leaves.sort_by_key(|(leaf, _, _)| *leaf);
    condition
}

/// Whether two fact lists, each sorted by variable, assign some variable
/// different values.
fn conflict(a: &[Fact], b: &[Fact]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].var.cmp(&b[j].var) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                if a[i].value != b[j].value {
                    return true;
                }
                i += 1;
                j += 1;
            }
        }
    }

    false
}

/// A value as an index.
fn widen(value: Value) -> usize {
    usize::try_from(value).expect("values fit usize")
}
