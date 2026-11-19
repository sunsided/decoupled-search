//! Uniform-cost decoupled search with dominance pruning.
//!
//! ```mermaid
//! graph LR
//!     pop[pop lowest g] --> stop{g >= best?}
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
use crate::{Error, Factoring, Plan};

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
#[derive(Debug, Default)]
struct Search {
    /// Every node that survived pruning, indexed by node id.
    nodes: Vec<Node>,
    /// Node ids ordered by `g`.
    open: BinaryHeap<Reverse<(Price, usize)>>,
    /// Node ids per center state, for dominance checks.
    by_center: BTreeMap<Vec<Value>, Vec<usize>>,
    /// Cheapest solution cost so far and the node it ends in.
    best: Option<(Price, usize)>,
}

/// Finds a cheapest plan for `task` by decoupled search over `factoring`.
///
/// Returns `Ok(None)` if the task is unsolvable.
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
    let decoupled = DecoupledTask::new(task, factoring)?;
    let mut search = Search::default();
    search.insert(&decoupled, decoupled.initial(), 0, None);

    while let Some(Reverse((g, id))) = search.open.pop() {
        if search.best.is_some_and(|(cost, _)| g >= cost) {
            break;
        }

        for (action, state) in decoupled.successors(&search.nodes[id].state) {
            let g = g.saturating_add(decoupled.globals[action].cost);
            search.insert(&decoupled, state, g, Some((id, action)));
        }
    }

    Ok(search
        .best
        .map(|(cost, id)| reconstruct::plan(&decoupled, &search.path(id), cost)))
}

impl Search {
    /// Adds `state` reached at cost `g` unless a stored node dominates it,
    /// and records it as the best solution if it is one.
    fn insert(
        &mut self,
        task: &DecoupledTask,
        state: DecoupledState,
        g: Price,
        parent: Option<(usize, usize)>,
    ) {
        if self.is_dominated(&state, g) {
            return;
        }

        let id = self.nodes.len();
        if let Some(rest) = task.goal_price(&state) {
            let cost = g.saturating_add(rest);
            if self.best.is_none_or(|(best, _)| cost < best) {
                self.best = Some((cost, id));
            }
        }

        self.by_center
            .entry(state.center.clone())
            .or_default()
            .push(id);
        self.open.push(Reverse((g, id)));
        self.nodes.push(Node { state, g, parent });
    }

    /// Whether a stored node has the same center, no higher `g` and no higher
    /// price anywhere. Such a node reaches every goal `state` reaches, at no
    /// higher cost.
    ///
    /// shortcut: linear scan per center state; replace with leaf-state
    /// dominance (design milestone 2) when stores get long.
    fn is_dominated(&self, state: &DecoupledState, g: Price) -> bool {
        self.by_center.get(&state.center).is_some_and(|ids| {
            ids.iter().any(|&id| {
                let node = &self.nodes[id];
                node.g <= g && node.state.prices_at_most(state)
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
