//! Uniform-cost, A* and greedy best-first decoupled search with dominance
//! pruning.
//!
//! All three share one driver and differ in the open-list key: `g`,
//! `g + h` or `h`. Uniform-cost search and A* stop once the popped key
//! reaches the best solution cost found so far; greedy search stops at the
//! first solution. [`Config`] selects the dominance criterion (see
//! [`Pruning`]) and whether leaves keep prices ([`Leaves`]).
//!
//! ```mermaid
//! graph LR
//!     pop[pop lowest key] --> stop{done?}
//!     stop -- yes --> done[reconstruct best]
//!     stop -- no --> expand[global successors]
//!     expand --> prune{dominated or dead end?}
//!     prune -- no --> store[store, record goal cost, push]
//!     store --> pop
//!     prune -- yes --> pop
//! ```

mod dominance;
mod reconstruct;

use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Task, Value};

use crate::decoupled::{DecoupledState, DecoupledTask, Price};
use crate::{DecoupledHeuristic, Error, Factoring, Plan};
use dominance::{Bounds, Dominance};

/// How the open list is ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// By `g + h`, cost-optimal for admissible `h` (`h = 0` without a
    /// heuristic, which is uniform-cost search).
    Cost,
    /// By `h` alone, stopping at the first solution.
    Greedy,
}

/// The dominance criterion that prunes generated decoupled states.
///
/// Every criterion preserves optimality of uniform-cost search and A*, and
/// completeness of greedy search; the dominance module's documentation
/// proves it for this crate's generalised global actions (see
/// `docs/design.md`). Following Torralba, Gnad, Dubbert and Hoffmann
/// (2016), [`Pruning::Frontier`] subsumes [`Pruning::Basic`],
/// [`Pruning::EffectivePrices`] subsumes [`Pruning::Frontier`],
/// [`Pruning::Simulation`] subsumes [`Pruning::Basic`], and
/// [`Pruning::EffectiveSimulation`] subsumes all; effective prices and
/// simulation are incomparable. Stronger criteria prune more but cost more
/// per check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Pruning {
    /// Same center and no higher price for any leaf state (Gnad and
    /// Hoffmann's criterion).
    Basic,
    /// Prices compared only on the frontier of the generated state: leaf
    /// states that are goals, satisfy a global precondition, or could still
    /// lower another state's price.
    Frontier,
    /// Prices of the stored state replaced by effective prices, which credit
    /// a leaf state with the cheapest way to get to where it leads.
    EffectivePrices,
    /// A leaf state may take the price of any reached leaf state that
    /// simulates it, under a leaf simulation relation.
    Simulation,
    /// Effective prices over simulation bounds; the strongest criterion.
    #[default]
    EffectiveSimulation,
}

/// What a decoupled state keeps per leaf state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Leaves {
    /// The price of every reached leaf state: the cheapest cost of leaf
    /// actions that reaches it along the current center path. Needed for
    /// cost-optimal search.
    #[default]
    Priced,
    /// Only which leaf states are reached (all at price 0); leaf action
    /// costs enter the plan, not the search. Smaller states and more
    /// pruning, for greedy search only.
    Reachability,
}

/// Options for decoupled search.
///
/// # Examples
///
/// ```
/// use decoupled_search::{Config, Factoring, Pruning};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// let op = |pre: Vec<Fact>, eff: Fact| Operator {
///     name: "op".into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let task = Task {
///     variables: vec![Variable::new("truck", 2), Variable::new("pkg", 3)],
///     operators: vec![
///         op(vec![Fact::new(0, 0)], Fact::new(0, 1)),
///         op(vec![Fact::new(0, 1)], Fact::new(0, 0)),
///         op(vec![Fact::new(0, 0), Fact::new(1, 0)], Fact::new(1, 2)),
///         op(vec![Fact::new(0, 1), Fact::new(1, 2)], Fact::new(1, 1)),
///     ],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(1, 1)],
///     axioms: vec![],
/// };
///
/// let config = Config {
///     pruning: Pruning::Basic,
///     ..Config::default()
/// };
/// let outcome = config.search(&task, &Factoring::fork(&task))?;
/// assert_eq!(outcome.plan.map(|p| p.cost), Some(3));
/// assert!(outcome.statistics.expanded >= 1);
/// # Ok::<(), decoupled_search::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The dominance criterion. Default: [`Pruning::EffectiveSimulation`].
    pub pruning: Pruning,
    /// Whether a stored state may dominate one reached more cheaply, when
    /// its cheaper prices pay for the difference in `g` (one budget shared
    /// by all leaves). Without it, the stored state needs no higher `g`.
    /// Default: `true`.
    pub fold_g: bool,
    /// Prices or reachability only. Default: [`Leaves::Priced`].
    pub leaves: Leaves,
    /// The largest leaf (in leaf states) that frontier, effective-price and
    /// simulation analyses run on; larger leaves use the basic comparison.
    /// Simulation takes time quadratic in the leaf size per refinement
    /// round, effective prices keep one bound per leaf state in every
    /// stored state. Default: 256.
    pub max_analysis_states: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pruning: Pruning::default(),
            fold_g: true,
            leaves: Leaves::default(),
            max_analysis_states: 256,
        }
    }
}

/// Counters of one search run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Statistics {
    /// Decoupled states expanded (popped and not past the stopping key).
    pub expanded: u64,
    /// Successors generated, the initial state included.
    pub generated: u64,
    /// Generated states pruned because a stored state dominates them.
    pub dominated: u64,
    /// Generated states pruned because the heuristic declared them dead
    /// ends.
    pub dead_ends: u64,
    /// States stored (and pushed on the open list).
    pub stored: u64,
}

/// The result of a search run: the plan, if any, and counters.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Outcome {
    /// The plan; `None` if the task is unsolvable.
    pub plan: Option<Plan>,
    /// What the search did.
    pub statistics: Statistics,
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
    /// The node's dominance bounds, computed on its first comparison.
    bounds: Option<Bounds>,
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
    /// The dominance criterion.
    dominance: Dominance,
    /// Every node that survived pruning, indexed by node id.
    nodes: Vec<Node>,
    /// Node ids ordered by their key (`g + h` or `h`).
    open: BinaryHeap<Reverse<(Price, usize)>>,
    /// Node ids per center state, for dominance checks.
    by_center: BTreeMap<Vec<Value>, Vec<usize>>,
    /// Cheapest solution cost so far and the node it ends in.
    best: Option<(Price, usize)>,
    /// Counters.
    statistics: Statistics,
}

/// Finds a cheapest plan for `task` by uniform-cost decoupled search over
/// `factoring`, with the default [`Config`].
///
/// Returns `Ok(None)` if the task is unsolvable. A generated state is pruned
/// when a stored state with the same center dominates it (see [`Pruning`]).
///
/// # Errors
///
/// - [`Error::InvalidTask`] if `task` does not validate.
/// - [`Error::CenterEffectReadsLeaf`], [`Error::EffectConditionAcrossLeaves`]
///   or [`Error::AxiomAcrossLeaves`] for conditional effects and axioms that
///   decoupled search cannot express under `factoring`.
/// - [`Error::DerivedInLeaf`] if `factoring` puts a derived variable in a
///   leaf.
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
/// assert_eq!(plan.validate(&task), Ok(()));
/// # Ok::<(), decoupled_search::Error>(())
/// ```
pub fn search(task: &Task, factoring: &Factoring) -> Result<Option<Plan>, Error> {
    Config::default()
        .search(task, factoring)
        .map(|outcome| outcome.plan)
}

/// Finds a cheapest plan for `task` by decoupled A* over `factoring`, guided
/// by `heuristic`, with the default [`Config`].
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
    heuristic: &(impl DecoupledHeuristic + ?Sized),
) -> Result<Option<Plan>, Error> {
    Config::default()
        .astar(task, factoring, heuristic)
        .map(|outcome| outcome.plan)
}

/// Finds some plan for `task` by decoupled greedy best-first search over
/// `factoring`, ordered by `heuristic` alone, with the default [`Config`],
/// and returns the first one found. The plan is valid but not necessarily
/// cheapest.
///
/// Dominance ignores costs here: a stored state dominates a generated one
/// if it can do everything the generated one can, which preserves
/// solvability. Returns `Ok(None)` if the task is unsolvable.
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
    heuristic: &(impl DecoupledHeuristic + ?Sized),
) -> Result<Option<Plan>, Error> {
    Config::default()
        .gbfs(task, factoring, heuristic)
        .map(|outcome| outcome.plan)
}

impl Config {
    /// Uniform-cost decoupled search with this configuration; see
    /// [`search()`].
    ///
    /// # Errors
    ///
    /// As [`search()`], and [`Error::ReachabilityNotOptimal`] with
    /// [`Leaves::Reachability`].
    pub fn search(&self, task: &Task, factoring: &Factoring) -> Result<Outcome, Error> {
        self.run(task, factoring, None, Order::Cost)
    }

    /// Decoupled A* with this configuration; see [`astar`].
    ///
    /// # Errors
    ///
    /// As [`Config::search`].
    pub fn astar(
        &self,
        task: &Task,
        factoring: &Factoring,
        heuristic: &(impl DecoupledHeuristic + ?Sized),
    ) -> Result<Outcome, Error> {
        self.run(task, factoring, Some(&heuristic), Order::Cost)
    }

    /// Decoupled greedy best-first search with this configuration; see
    /// [`gbfs`]. With [`Leaves::Reachability`], states keep no prices.
    ///
    /// # Errors
    ///
    /// As [`search()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Config, Factoring, HFf, Leaves};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// let op = |pre: Vec<Fact>, eff: Fact, cost| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("truck", 2), Variable::new("pkg", 3)],
    ///     operators: vec![
    ///         op(vec![Fact::new(0, 0)], Fact::new(0, 1), 4),
    ///         op(vec![Fact::new(0, 0), Fact::new(1, 0)], Fact::new(1, 2), 2),
    ///         op(vec![Fact::new(0, 1), Fact::new(1, 2)], Fact::new(1, 1), 3),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let config = Config { leaves: Leaves::Reachability, ..Config::default() };
    /// let outcome = config.gbfs(&task, &Factoring::fork(&task), &HFf::new(&task))?;
    /// let plan = outcome.plan.expect("solvable");
    /// assert_eq!(plan.cost, 9, "costs are summed from the plan's operators");
    /// assert_eq!(plan.validate(&task), Ok(()));
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    pub fn gbfs(
        &self,
        task: &Task,
        factoring: &Factoring,
        heuristic: &(impl DecoupledHeuristic + ?Sized),
    ) -> Result<Outcome, Error> {
        self.run(task, factoring, Some(&heuristic), Order::Greedy)
    }

    /// The search driver shared by all entry points.
    ///
    /// # Errors
    ///
    /// As [`Config::search`].
    fn run(
        &self,
        task: &Task,
        factoring: &Factoring,
        heuristic: Option<&dyn DecoupledHeuristic>,
        order: Order,
    ) -> Result<Outcome, Error> {
        let mut decoupled = DecoupledTask::new(task, factoring)?;
        if self.leaves == Leaves::Reachability {
            if order != Order::Greedy {
                return Err(Error::ReachabilityNotOptimal);
            }

            decoupled = decoupled.into_reachability_only();
        }

        let mut search = Search {
            task: &decoupled,
            heuristic,
            order,
            dominance: Dominance::new(&decoupled, self, order == Order::Greedy),
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            by_center: BTreeMap::new(),
            best: None,
            statistics: Statistics::default(),
        };
        search.run();

        let plan = search.best.map(|(cost, id)| {
            let plan = reconstruct::plan(&decoupled, &search.path(id), cost);
            with_true_cost(task, plan, decoupled.is_reachability_only())
        });

        Ok(Outcome {
            plan,
            statistics: search.statistics,
        })
    }
}

/// `plan` with its cost summed from its operators. Without leaf costs in
/// the search this is the only correct cost; otherwise it equals the search
/// cost, which a debug assertion checks.
fn with_true_cost(task: &Task, mut plan: Plan, reachability_only: bool) -> Plan {
    let summed = plan
        .ops
        .iter()
        .map(|&op| u64::from(task.operators[op].cost))
        .sum();
    debug_assert!(
        reachability_only || summed == plan.cost,
        "search cost {} differs from plan cost {summed}",
        plan.cost
    );
    plan.cost = summed;
    plan
}

impl Search<'_> {
    /// Inserts the initial state and expands nodes until the open list is
    /// empty or [`Search::is_done`].
    fn run(&mut self) {
        self.insert(self.task.initial(), 0, None);

        while let Some(Reverse((key, id))) = self.open.pop() {
            if self.is_done(key) {
                break;
            }

            self.statistics.expanded += 1;
            let g = self.nodes[id].g;
            for (action, state) in self.task.successors(&self.nodes[id].state) {
                let g = g.saturating_add(self.task.globals[action].cost);
                self.insert(state, g, Some((id, action)));
            }
        }
    }

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
        self.statistics.generated += 1;
        if self.is_dominated(&state, g) {
            self.statistics.dominated += 1;
            return;
        }

        let Some(h) = self.estimate(&state) else {
            self.statistics.dead_ends += 1;
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
        self.nodes.push(Node {
            state,
            g,
            parent,
            bounds: None,
        });
        self.statistics.stored += 1;
    }

    /// Whether a stored node with the same center dominates `state`
    /// reached at `g`, under the configured criterion.
    ///
    /// shortcut: linear scan per center state; index the store (for
    /// example by reached-leaf-state sets) when stores get long.
    fn is_dominated(&mut self, state: &DecoupledState, g: Price) -> bool {
        let Some(ids) = self.by_center.get(&state.center) else {
            return false;
        };

        let frontier = self.dominance.frontier(self.task, state);
        for &id in ids {
            let node = &mut self.nodes[id];
            let bounds = node
                .bounds
                .get_or_insert_with(|| self.dominance.bounds(self.task, &node.state));
            if self
                .dominance
                .dominates((&node.state, node.g, bounds), (state, g, &frontier))
            {
                return true;
            }
        }

        false
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
