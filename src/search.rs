//! Uniform-cost, A* and greedy best-first decoupled search with dominance
//! pruning.
//!
//! All three share one driver and differ in the open-list key: `g`,
//! `g + h` or `h`. Uniform-cost search and A* stop once the popped key
//! reaches the best solution cost found so far; greedy search stops at the
//! first solution.
//!
//! ```mermaid
//! graph LR
//!     pop[pop lowest key] --> stop{done?}
//!     stop -- yes --> done[reconstruct best]
//!     stop -- no --> expand[global successors]
//!     expand --> prune{dominated?}
//!     prune -- no --> store[store, record goal cost, push]
//!     store --> pop
//!     prune -- yes --> pop
//! ```

mod reconstruct;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Task, Value};

use crate::decoupled::{DecoupledState, DecoupledTask, Price};
use crate::{DecoupledHeuristic, Error, Factoring, Plan};

/// How the open list is ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// By `g + h`, cost-optimal for admissible `h` (`h = 0` without a
    /// heuristic, which is uniform-cost search).
    Cost,
    /// By `h` alone, stopping at the first solution.
    Greedy,
}

/// A generated decoupled state.
#[derive(Debug)]
struct Node {
    /// The decoupled state.
    state: DecoupledState,
    /// Cost of the global actions from the initial state.
    g: Price,
    /// Predecessor node and the index of the global action that led here;
    /// `None` for the initial node.
    parent: Option<(usize, usize)>,
}

/// Search bookkeeping: all nodes, the open list, the dominance store and the
/// best solution found so far.
struct Search<'t> {
    /// The task being searched.
    task: &'t DecoupledTask,
    /// The heuristic; `None` means `h = 0` without computing fact prices.
    heuristic: Option<&'t dyn DecoupledHeuristic>,
    /// The open-list order.
    order: Order,
    /// Every node that survived pruning, indexed by node id.
    nodes: Vec<Node>,
    /// Node ids ordered by their key (`g + h` or `h`).
    open: BinaryHeap<Reverse<(Price, usize)>>,
    /// Node ids per center state, for dominance checks.
    by_center: BTreeMap<Vec<Value>, Vec<usize>>,
    /// Cheapest solution cost so far and the node it ends in.
    best: Option<(Price, usize)>,
}

/// Finds a cheapest plan for `task` by uniform-cost decoupled search over
/// `factoring`.
///
/// Returns `Ok(None)` if the task is unsolvable. A generated state is pruned
/// when a stored state with the same center has no higher `g` and no higher
/// price for any leaf state.
///
/// # Errors
///
/// - [`Error::InvalidTask`] if `task` does not validate.
/// - [`Error::Axioms`] if `task` has axioms or derived variables.
/// - [`Error::ConditionalEffects`] if an operator has conditional effects.
/// - [`Error::FactoringMismatch`] or [`Error::LeafTooLarge`] if `factoring`
///   was built for a different task.
///
/// # Examples
///
/// ```
/// use decoupled_search::{Factoring, search};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // A truck (variable 0) and a package (variable 1) that rides along.
/// let op = |name: &str, pre: Vec<Fact>, eff: Fact| Operator {
///     name: name.into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let task = Task {
///     variables: vec![Variable::new("truck", 2), Variable::new("pkg", 3)],
///     operators: vec![
///         op("drive", vec![Fact::new(0, 0)], Fact::new(0, 1)),
///         op("load", vec![Fact::new(0, 0), Fact::new(1, 0)], Fact::new(1, 2)),
///         op("unload", vec![Fact::new(0, 1), Fact::new(1, 2)], Fact::new(1, 1)),
///     ],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(1, 1)],
///     axioms: vec![],
/// };
///
/// let factoring = Factoring::fork(&task);
/// assert_eq!(factoring.leaves(), &[vec![1]]);
///
/// let plan = search(&task, &factoring)?.expect("solvable");
/// assert_eq!(plan.cost, 3);
/// assert_eq!(task.validate_plan(&plan.ops), Some(3));
/// # Ok::<(), decoupled_search::Error>(())
/// ```
pub fn search(task: &Task, factoring: &Factoring) -> Result<Option<Plan>, Error> {
    run(task, factoring, None, Order::Cost)
}

/// Finds a cheapest plan for `task` by decoupled A* over `factoring`, guided
/// by `heuristic`.
///
/// The plan is cost-optimal if `heuristic` is admissible in the sense of
/// [`DecoupledHeuristic::estimate`], as [`crate::HMax`] is. Pruning is the
/// same as in [`search()`]; states the heuristic declares dead ends are
/// pruned too. Returns `Ok(None)` if the task is unsolvable.
///
/// # Errors
///
/// As [`search()`].
///
/// # Examples
///
/// See [`crate::HMax`].
pub fn astar(
    task: &Task,
    factoring: &Factoring,
    heuristic: &impl DecoupledHeuristic,
) -> Result<Option<Plan>, Error> {
    run(task, factoring, Some(heuristic), Order::Cost)
}

/// Finds some plan for `task` by decoupled greedy best-first search over
/// `factoring`, ordered by `heuristic` alone, and returns the first one
/// found. The plan is valid but not necessarily cheapest.
///
/// A generated state is pruned when a stored state with the same center has
/// no higher price for any leaf state, whatever its `g`: that state reaches
/// every goal the pruned one reaches, so solvability is preserved. Returns
/// `Ok(None)` if the task is unsolvable.
///
/// # Errors
///
/// As [`search()`].
///
/// # Examples
///
/// See [`crate::HFf`].
pub fn gbfs(
    task: &Task,
    factoring: &Factoring,
    heuristic: &impl DecoupledHeuristic,
) -> Result<Option<Plan>, Error> {
    run(task, factoring, Some(heuristic), Order::Greedy)
}

/// The search driver shared by [`search()`], [`astar`] and [`gbfs`].
///
/// # Errors
///
/// As [`search()`].
fn run(
    task: &Task,
    factoring: &Factoring,
    heuristic: Option<&dyn DecoupledHeuristic>,
    order: Order,
) -> Result<Option<Plan>, Error> {
    let decoupled = DecoupledTask::new(task, factoring)?;
    let mut search = Search {
        task: &decoupled,
        heuristic,
        order,
        nodes: Vec::new(),
        open: BinaryHeap::new(),
        by_center: BTreeMap::new(),
        best: None,
    };
    search.insert(decoupled.initial(), 0, None);

    while let Some(Reverse((key, id))) = search.open.pop() {
        if search.is_done(key) {
            break;
        }

        let g = search.nodes[id].g;
        for (action, state) in decoupled.successors(&search.nodes[id].state) {
            let g = g.saturating_add(decoupled.globals[action].cost);
            search.insert(state, g, Some((id, action)));
        }
    }

    Ok(search
        .best
        .map(|(cost, id)| reconstruct::plan(&decoupled, &search.path(id), cost)))
}

impl Search<'_> {
    /// Whether the search can stop before expanding a node with open-list
    /// key `key`: greedy search once any solution is known, otherwise once
    /// `key` reaches the best solution cost (keys are lower bounds).
    fn is_done(&self, key: Price) -> bool {
        match self.order {
            Order::Cost => self.best.is_some_and(|(cost, _)| key >= cost),
            Order::Greedy => self.best.is_some(),
        }
    }

    /// The heuristic estimate of `state`; 0 without a heuristic, `None` for
    /// a dead end.
    fn estimate(&self, state: &DecoupledState) -> Option<Price> {
        self.heuristic
            .map_or(Some(0), |h| h.estimate(&self.task.fact_prices(state)))
    }

    /// Adds `state` reached at cost `g` unless a stored node dominates it or
    /// the heuristic declares it a dead end, and records it as the best
    /// solution if it is one.
    fn insert(&mut self, state: DecoupledState, g: Price, parent: Option<(usize, usize)>) {
        if self.is_dominated(&state, g) {
            return;
        }

        let Some(h) = self.estimate(&state) else {
            return;
        };

        let id = self.nodes.len();
        if let Some(rest) = self.task.goal_price(&state) {
            let cost = g.saturating_add(rest);
            if self.best.is_none_or(|(best, _)| cost < best) {
                self.best = Some((cost, id));
            }
        }

        self.by_center
            .entry(state.center.clone())
            .or_default()
            .push(id);
        let key = match self.order {
            Order::Cost => g.saturating_add(h),
            Order::Greedy => h,
        };
        self.open.push(Reverse((key, id)));
        self.nodes.push(Node { state, g, parent });
    }

    /// Whether a stored node has the same center, no higher price anywhere
    /// and, unless the search is greedy, no higher `g`. Such a node reaches
    /// every goal `state` reaches, and at no higher cost if `g` is compared.
    /// Leaf prices exclude `g`, so comparing prices alone would lose
    /// optimality.
    ///
    /// shortcut: linear scan per center state; replace with leaf-state
    /// dominance (design milestone 2) when stores get long.
    fn is_dominated(&self, state: &DecoupledState, g: Price) -> bool {
        self.by_center.get(&state.center).is_some_and(|ids| {
            ids.iter().any(|&id| {
                let node = &self.nodes[id];
                (self.order == Order::Greedy || node.g <= g) && node.state.prices_at_most(state)
            })
        })
    }

    /// The path from the initial node to node `id`, as
    /// `(global action index that led here, state)`, initial node first.
    fn path(&self, mut id: usize) -> Vec<(Option<usize>, &DecoupledState)> {
        let mut path = Vec::new();
        loop {
            let node = &self.nodes[id];
            path.push((node.parent.map(|(_, a)| a), &node.state));
            match node.parent {
                Some((parent, _)) => id = parent,
                None => break,
            }
        }

        path.reverse();
        path
    }
}
