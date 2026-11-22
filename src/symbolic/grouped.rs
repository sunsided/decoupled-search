//! Decoupled search with a symbolic center ([`crate::Config::symbolic_center`]):
//! a node is a set of center states, kept as a BDD, that share one pricing
//! function and one `g`, so a global action is applied to all of them at
//! once. This is this crate's formulation; see `docs/design.md` for why
//! each step keeps the decoupled state space's semantics.
//!
//! - *Members.* Node `(C, P, g)` stands for the decoupled states
//!   `(c, P)` reached at cost `g`, for every `c` in `C`.
//! - *Expansion.* For global action `a`, the members whose center
//!   satisfies `a`'s center preconditions move to the image of `C` under
//!   `a`'s center relation, with the pricing carried across `a`'s leaf
//!   parts. The leaf closure depends on the center only through which leaf
//!   actions' center preconditions hold, so the image is split into
//!   classes that agree on all of them, and each class is closed once.
//! - *Merging.* A new node joins a stored, unexpanded one with the same
//!   `g` and the same pricing (canonical BDDs compare by value), created
//!   after its parent, so parent links always point to older nodes.
//! - *Dominance.* A stored node with no higher `g` and pointwise no higher
//!   prices removes its center states from the new node
//!   ([`Pruning::Basic`] per member); the node is pruned once none remain.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use alloc::collections::BinaryHeap;
use sas_plus::{Task, Value};
use symbolic_search::backend::{Backend, BackendResult};

use super::bdd_leaf::{Encoding, meets, minus};
use super::{Leaf, Pricing, Set, SymbolicTask, widen};
use crate::decoupled::Price;
use crate::search::{Order, with_true_cost};
use crate::{Config, DecoupledHeuristic, Outcome, Pruning, Statistics};

/// The center's compiled parts.
struct Center<'t> {
    /// The encoding.
    encoding: &'t Encoding,
    /// Each global action's center relation.
    relations: Vec<Set>,
    /// The center states satisfying each global action's center
    /// preconditions.
    pres: Vec<Set>,
    /// The center goal.
    goal: Set,
    /// The distinct center preconditions of leaf actions, with their sets.
    conditions: Vec<(Vec<(usize, Value)>, Set)>,
    /// The center states with each slot's value.
    facts: Vec<Vec<Set>>,
}

/// A search node.
struct Node {
    /// Its center states.
    center: Set,
    /// The shared pricing.
    leaves: Vec<Pricing>,
    /// The shared cost.
    g: Price,
    /// The current open-list key.
    key: Price,
    /// Whether it was expanded.
    expanded: bool,
    /// Where its center states came from: parent node, global action, and
    /// the center states it contributed.
    sources: Vec<(usize, usize, Set)>,
}

/// Best-first search over nodes with symbolic centers.
pub(super) fn search(
    task: &SymbolicTask,
    original: &Task,
    config: &Config,
    heuristic: Option<&dyn DecoupledHeuristic>,
    order: Order,
) -> BackendResult<Outcome> {
    debug_assert_eq!(config.pruning, Pruning::Basic);
    let center = Center::new(task)?;
    let mut search = Grouped {
        task,
        center,
        heuristic,
        greedy: order == Order::Greedy,
        nodes: Vec::new(),
        open: BinaryHeap::new(),
        best: None,
        statistics: Statistics::default(),
    };
    search.run()?;

    let plan = match search.best {
        None => None,
        Some((cost, id)) => Some(with_true_cost(
            original,
            search.plan(id, cost)?,
            !task.costs,
        )),
    };

    Ok(Outcome {
        plan,
        statistics: search.statistics,
    })
}

impl<'t> Center<'t> {
    /// The center parts of `task`.
    fn new(task: &'t SymbolicTask) -> BackendResult<Self> {
        let bdd = &task.bdd;
        let encoding = task
            .center_encoding
            .as_ref()
            .expect("a symbolic center is encoded");
        let cube = |facts: &[(usize, Value)]| -> BackendResult<Set> {
            let mut set = bdd.constant(true);
            for &(slot, value) in facts {
                set = bdd.and(&set, &encoding.value(bdd, slot, widen(value), 0)?)?;
            }
            Ok(set)
        };
        let local = |facts: &[(usize, Value)]| {
            facts
                .iter()
                .map(|&(s, v)| (s, widen(v)))
                .collect::<Vec<_>>()
        };

        let mut relations = Vec::new();
        let mut pres = Vec::new();
        for g in &task.globals {
            relations.push(encoding.relation(bdd, &local(&g.center_pre), &local(&g.center_eff))?);
            pres.push(cube(&g.center_pre)?);
        }

        let mut conditions: Vec<(Vec<(usize, Value)>, Set)> = Vec::new();
        for leaf in &task.leaves {
            let pres: Vec<Vec<(usize, Value)>> = match leaf {
                Leaf::Symbolic(l) => l.conditions().map(<[_]>::to_vec).collect(),
                Leaf::Explicit(l) => l.moves.iter().map(|m| m.center_pre.clone()).collect(),
            };
            for pre in pres {
                if !pre.is_empty() && conditions.iter().all(|(p, _)| *p != pre) {
                    let set = cube(&pre)?;
                    conditions.push((pre, set));
                }
            }
        }

        let facts = task
            .center_domains
            .iter()
            .enumerate()
            .map(|(slot, &d)| (0..d).map(|v| encoding.value(bdd, slot, v, 0)).collect())
            .collect::<BackendResult<Vec<Vec<Set>>>>()?;

        Ok(Self {
            encoding,
            relations,
            pres,
            goal: cube(&task.center_goal)?,
            conditions,
            facts,
        })
    }

    /// `set` split into classes on which every leaf-action center
    /// precondition is either true throughout or false throughout, each
    /// with which of [`Center::conditions`] hold.
    fn classes(&self, task: &SymbolicTask, set: Set) -> BackendResult<Vec<(Set, Vec<bool>)>> {
        let bdd = &task.bdd;
        let mut classes = vec![(set, Vec::new())];
        for (_, condition) in &self.conditions {
            let mut next = Vec::with_capacity(classes.len() * 2);
            for (class, signature) in classes {
                let inside = bdd.and(&class, condition)?;
                let outside = minus(bdd, &class, condition)?;
                for (part, holds) in [(inside, true), (outside, false)] {
                    if !bdd.is_empty(&part) {
                        let mut s = signature.clone();
                        s.push(holds);
                        next.push((part, s));
                    }
                }
            }
            classes = next;
        }

        Ok(classes)
    }
}

/// Search bookkeeping.
struct Grouped<'t> {
    /// The task.
    task: &'t SymbolicTask,
    /// Its center parts.
    center: Center<'t>,
    /// The heuristic; `None` for `h = 0`.
    heuristic: Option<&'t dyn DecoupledHeuristic>,
    /// Greedy search: order by `h`, dominance ignores `g`.
    greedy: bool,
    /// Every stored node.
    nodes: Vec<Node>,
    /// `(key, node)`; stale entries are skipped.
    open: BinaryHeap<Reverse<(Price, usize)>>,
    /// Cheapest solution so far and its node.
    best: Option<(Price, usize)>,
    /// Counters.
    statistics: Statistics,
}

impl Grouped<'_> {
    /// Inserts the initial node and expands until done.
    fn run(&mut self) -> BackendResult<()> {
        let task = self.task;
        let bdd = &task.bdd;
        let mut center = bdd.constant(true);
        for (slot, &value) in task.initial_center.iter().enumerate() {
            center = bdd.and(&center, &self.center.facts[slot][widen(value)])?;
        }
        let seeds: Vec<Pricing> = (0..task.leaves.len())
            .map(|l| task.initial_seeds(l))
            .collect();
        let initial = &task.initial_center;
        let leaves = task.closed(&seeds, &|pre| pre.iter().all(|&(s, v)| initial[s] == v))?;
        self.insert(center, leaves, 0, None)?;

        while let Some(Reverse((key, id))) = self.open.pop() {
            if self.nodes[id].expanded || self.nodes[id].key != key {
                continue;
            }
            let done = if self.greedy {
                self.best.is_some()
            } else {
                self.best.is_some_and(|(c, _)| key >= c)
            };
            if done {
                break;
            }

            self.nodes[id].expanded = true;
            self.statistics.expanded += 1;
            self.expand(id)?;
        }

        Ok(())
    }

    /// Applies every global action to the members of node `id`.
    fn expand(&mut self, id: usize) -> BackendResult<()> {
        let task = self.task;
        let bdd = &task.bdd;
        for a in 0..task.globals.len() {
            let applicable = bdd.and(&self.nodes[id].center, &self.center.pres[a])?;
            if bdd.is_empty(&applicable) {
                continue;
            }
            let Some(seeds) = task.carried(a, &self.nodes[id].leaves)? else {
                continue;
            };

            let image = self
                .center
                .encoding
                .image(bdd, &applicable, &self.center.relations[a])?;
            let g = self.nodes[id].g.saturating_add(task.globals[a].cost);
            for (class, signature) in self.center.classes(task, image)? {
                let conditions = &self.center.conditions;
                let active = |pre: &[(usize, Value)]| {
                    pre.is_empty()
                        || conditions
                            .iter()
                            .position(|(p, _)| p == pre)
                            .is_some_and(|i| signature[i])
                };
                let leaves = task.closed(&seeds, &active)?;
                self.insert(class, leaves, g, Some((id, a)))?;
            }
        }

        Ok(())
    }

    /// The center states of `center` that no stored node dominates at `g`
    /// with `leaves`.
    fn undominated(&self, mut center: Set, leaves: &[Pricing], g: Price) -> BackendResult<Set> {
        let bdd = &self.task.bdd;
        for node in &self.nodes {
            if !meets(bdd, &node.center, &center)? {
                continue;
            }
            if self
                .task
                .dominates((&node.leaves, node.g), (leaves, g), self.greedy, false)?
            {
                center = minus(bdd, &center, &node.center)?;
                if bdd.is_empty(&center) {
                    break;
                }
            }
        }

        Ok(center)
    }

    /// The heuristic value of members `center` with `leaves`; `None` for a
    /// dead end. Center facts count as reached if some member has them.
    fn estimate(&self, center: &Set, leaves: &[Pricing]) -> BackendResult<Option<Price>> {
        let Some(h) = self.heuristic else {
            return Ok(Some(0));
        };
        let bdd = &self.task.bdd;
        let facts = &self.center.facts;
        let prices = self.task.fact_prices(
            &|slot, value| meets(bdd, center, &facts[slot][value]),
            leaves,
        )?;
        Ok(h.estimate(&prices))
    }

    /// Records node `id` as the best solution if it contains a goal state
    /// cheaper than the best so far.
    fn record_goal(&mut self, id: usize) -> BackendResult<()> {
        let node = &self.nodes[id];
        if !meets(&self.task.bdd, &node.center, &self.center.goal)? {
            return Ok(());
        }
        if let Some(rest) = self.task.leaves_goal_price(&node.leaves)? {
            let cost = node.g.saturating_add(rest);
            if self.best.is_none_or(|(b, _)| cost < b) {
                self.best = Some((cost, id));
            }
        }

        Ok(())
    }

    /// Adds center states `center` with `leaves` at `g`, reached from
    /// `parent` (node and global action): pruned where dominated, merged
    /// into a matching open node, or stored as a new node.
    fn insert(
        &mut self,
        center: Set,
        leaves: Vec<Pricing>,
        g: Price,
        parent: Option<(usize, usize)>,
    ) -> BackendResult<()> {
        self.statistics.generated += 1;
        let bdd = &self.task.bdd;
        let center = self.undominated(center, &leaves, g)?;
        if bdd.is_empty(&center) {
            self.statistics.dominated += 1;
            return Ok(());
        }

        let source = |c: &Set| parent.map(|(p, a)| (p, a, c.clone()));
        let after = parent.map_or(0, |(p, _)| p + 1);
        let matching = (after..self.nodes.len()).find(|&u| {
            !self.nodes[u].expanded && self.nodes[u].g == g && self.nodes[u].leaves == leaves
        });
        if let Some(u) = matching {
            let merged = bdd.or(&self.nodes[u].center, &center)?;
            let Some(h) = self.estimate(&merged, &leaves)? else {
                return Ok(());
            };
            let key = if self.greedy { h } else { g.saturating_add(h) };
            let node = &mut self.nodes[u];
            node.center = merged;
            node.sources.extend(source(&center));
            if key != node.key {
                node.key = key;
                self.open.push(Reverse((key, u)));
            }
            return self.record_goal(u);
        }

        let Some(h) = self.estimate(&center, &leaves)? else {
            self.statistics.dead_ends += 1;
            return Ok(());
        };
        let key = if self.greedy { h } else { g.saturating_add(h) };
        let id = self.nodes.len();
        self.nodes.push(Node {
            sources: source(&center).into_iter().collect(),
            center,
            leaves,
            g,
            key,
            expanded: false,
        });
        self.open.push(Reverse((key, id)));
        self.statistics.stored += 1;
        self.record_goal(id)
    }

    /// The plan for the best solution, in node `id` at total `cost`: a
    /// concrete goal center state, walked back through the nodes' sources
    /// one center state at a time.
    fn plan(&self, id: usize, cost: Price) -> BackendResult<crate::Plan> {
        let bdd = &self.task.bdd;
        let encoding = self.center.encoding;
        let mut at = id;
        let mut center =
            encoding.pick(bdd, &bdd.and(&self.nodes[id].center, &self.center.goal)?)?;
        let mut stages: Vec<(Option<usize>, Set, usize)> = Vec::new();
        loop {
            let node = &self.nodes[at];
            let mut back = None;
            for (p, a, part) in &node.sources {
                if meets(bdd, part, &center)? {
                    back = Some((*p, *a));
                    break;
                }
            }
            stages.push((back.map(|(_, a)| a), center.clone(), at));
            let Some((p, a)) = back else { break };
            let before = encoding.preimage(bdd, &center, &self.center.relations[a])?;
            center = encoding.pick(bdd, &bdd.and(&before, &self.nodes[p].center)?)?;
            at = p;
        }
        stages.reverse();

        let centers: Vec<Vec<Value>> = stages
            .iter()
            .map(|(_, c, _)| {
                encoding
                    .decode(bdd, c)
                    .into_iter()
                    .map(|v| Value::try_from(v).expect("values fit u32"))
                    .collect()
            })
            .collect();
        let actions: Vec<Option<usize>> = stages.iter().map(|(a, ..)| *a).collect();
        let pricings: Vec<&[Pricing]> = stages
            .iter()
            .map(|(.., n)| self.nodes[*n].leaves.as_slice())
            .collect();
        self.task.plan(&centers, &actions, &pricings, cost)
    }
}
