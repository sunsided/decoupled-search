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
use dominance::{Bounds, Dominance, simulation};

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
/// A stored state `t` *dominates* a generated state `n` with the same
/// center if every completion of `n` can be turned into a completion of `t`
/// with the same remaining global actions and no higher total cost; `n` is
/// then pruned. The criteria are those of Torralba, Gnad, Dubbert and
/// Hoffmann (IJCAI 2016), generalised to this crate's global actions, which
/// may read and write leaves under any partition, and to partial leaf
/// goals. [`Pruning::Frontier`] subsumes [`Pruning::Basic`],
/// [`Pruning::EffectivePrices`] subsumes [`Pruning::Frontier`],
/// [`Pruning::Simulation`] subsumes [`Pruning::Basic`], and
/// [`Pruning::EffectiveSimulation`] subsumes all; effective prices and
/// simulation are incomparable (as in the paper). Stronger criteria prune
/// more but cost more: simulation is computed once per search and leaf.
///
/// # Anchored leaf states
///
/// The paper has one goal leaf state, and only leaf actions move a leaf.
/// Here, a leaf state is *anchored* if it satisfies the leaf goal or the
/// leaf preconditions of some global action's part on the leaf (every state
/// is, if some part has no leaf precondition). An unanchored state can only
/// be left by leaf actions; a completion's leaf path ends in a goal leaf
/// state and meets every global action that touches the leaf in an
/// anchored state.
///
/// # The per-leaf property
///
/// Each criterion yields, per leaf `L`, a slack `δ_L` such that:
///
/// > (★) for every center path `π` from `n` and every `π`-compliant leaf
/// > path `ρ` that a completion of `n` takes from a reached leaf state `x`,
/// > `t` has a `π`-compliant leaf path that ends in a goal leaf state and
/// > satisfies the same global preconditions, at total leaf cost at most
/// > `price_n(x) + δ_L + cost(ρ)`.
///
/// All criteria but the frontier one give a *bound* `D_t(x)` with "`t` can
/// do what `ρ` does from `x` at leaf cost at most `D_t(x) + cost(ρ)`"; then
/// `δ_L = max_x (D_t(x) - price_n(x))` over the reached states `x` of `n`
/// satisfies (★).
///
/// - [`Pruning::Basic`]: `D_t(x) = price_t(x)`. `t` reaches `x` at that
///   price and follows `ρ`.
/// - [`Pruning::Simulation`]: `D_t(x) = min_{x ⪯ y} price_t(y)` over the
///   reached `y` that simulate `x` (see the variant). `t` reaches `y` and
///   mirrors `ρ`.
/// - [`Pruning::EffectivePrices`]: `D_t = p`, the point-wise least function
///   with `p(x) = S(x)` on anchored states and
///   `p(x) = min(S(x), max_{x -a-> y} p(y) - cost(a))` elsewhere, over leaf
///   actions regardless of their center preconditions, where
///   `S = price_t`. Proof: let `ρ` visit `x = s_0, ..., s_m` and take the
///   first `i` with `p(s_i) = S(s_i)`; it exists because `ρ` ends in a goal
///   leaf state, which is anchored. Before `i` all states are unanchored,
///   so every step is a leaf action and `p(s_j) >= p(s_{j+1}) - cost(a_{j+1})`;
///   summing, `S(s_i) = p(s_i) <= p(x) + cost(s_0 .. s_i)`. `t` reaches `s_i`
///   at `S(s_i)` and follows the rest of `ρ`. States whose least value is
///   `-∞` reach no anchored state, so no completion uses them.
/// - [`Pruning::EffectiveSimulation`]: the same with `S` the simulation
///   bound; at `s_i`, `t` reaches a simulating `y` and mirrors the rest.
/// - [`Pruning::Frontier`]: `δ_L = max_x (price_t(x) - price_n(x))` over the
///   reached states `x` on the frontier of `n`: anchored, or left by some
///   leaf action `x -a-> y` with `price_n(x) + cost(a) < price_n(y)`. Proof:
///   if `ρ` starts outside the frontier, every leaf action out of a
///   non-frontier state `s_j` reaches `s_{j+1}` with
///   `price_n(s_{j+1}) <= price_n(s_j) + cost(a_{j+1})`, so the first
///   frontier state `s_k` of `ρ` (it exists, the goal state is anchored)
///   has `price_n(s_k) <= price_n(x) + cost(s_0 .. s_k)`; `t` reaches `s_k`
///   at `price_t(s_k) <= price_n(s_k) + δ_L` and follows the rest of `ρ`.
///
/// # The `g` budget
///
/// Leaf prices exclude `g`. A completion of `n` through leaf states `x_L`
/// costs `g(n) + Σ_L (price_n(x_L) + cost(ρ_L)) + c`, with `c` the cost of
/// the remaining global actions, and by (★) `t` has one costing at most
/// `g(t) + Σ_L (price_n(x_L) + δ_L + cost(ρ_L)) + c`. So `t` dominates `n`
/// for cost-optimal search if `g(t) + Σ_L δ_L <= g(n)`, which
/// [`Config::fold_g`] uses; without it, the classic condition `g(t) <= g(n)`
/// and `δ_L <= 0` for every leaf. Comparing `g + price` per leaf state
/// instead is unsound with two or more leaves, because `g` is paid once but
/// would be credited per leaf: with `g(t) = 0` and both leaves at price 10,
/// `g(n) = 10` and both at 0, every leaf ties at 10, yet completing both
/// costs 20 from `t` and 10 from `n`. Greedy search only needs
/// reachability: `t` dominates `n` if every `δ_L` is finite.
///
/// # Optimality
///
/// Say a stored node *covers* cost `C` via global actions `π` if `π` leads
/// it to a goal state with total cost at most `C`. While uniform-cost search
/// or A* with an admissible heuristic has not recorded the optimal cost
/// `C*`, take a node covering `C*` with the shortest `π`. If `π` is empty,
/// the node is a goal and its cost was recorded when it was stored. If the
/// node were closed, its successor along `π[0]` was either stored (covering
/// via a shorter path), pruned by a stored node covering via the same
/// shorter `π[1..]` (dominance keeps the remaining global actions, so the
/// paper's condition on the number of center actions is not needed), or a
/// dead end of the heuristic, which is impossible for a state with a
/// completion. So it is open with `f <= C*`, and the search cannot stop
/// before recording `C*`. Termination follows from Dickson's lemma for the
/// basic criterion, which every other one subsumes.
///
/// Leaves whose transitions depend on the center (center-conditioned
/// effects, derived variables attached to them) or that have more than
/// [`Config::max_analysis_states`] states use the basic bound.
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
    /// simulates it.
    ///
    /// The leaf simulation `⪯` (Definition 5 of the paper, with global
    /// actions added) is the coarsest relation on leaf states where
    /// `x ⪯ y` implies: if `x` satisfies the leaf goal, so does `y`; for
    /// every global action's part `P` on the leaf, if `x` satisfies `P`'s
    /// leaf preconditions, so does `y`, and `P(x) ⪯ P(y)`; and for every
    /// leaf action `x -a-> x'`, either `x' ⪯ y` (the leaf stays at `y`) or
    /// some leaf action `b` with `cost(b) <= cost(a)` and center
    /// preconditions contained in `a`'s leads `y -b-> y'` with `x' ⪯ y'`.
    /// By induction on a compliant leaf path from `x`, `y` has one that
    /// complies with the same center path (each `b` is applicable wherever
    /// `a` was, each global action's leaf preconditions hold), ends in a
    /// goal leaf state whenever the original does, and costs no more.
    Simulation,
    /// Effective prices over simulation bounds; the strongest criterion.
    EffectiveSimulation,
    /// [`Pruning::EffectivePrices`] until the search has spent about as
    /// much as computing the leaf simulations would cost, then
    /// [`Pruning::EffectiveSimulation`] (the default).
    ///
    /// Computing a leaf simulation takes time quadratic in the leaf size,
    /// which dominates searches on small tasks. The switch follows the
    /// ski-rental rule: the work unit is one leaf state of a generated
    /// decoupled state (leaf closure touches every leaf state of every
    /// successor), and the simulation's estimated cost is the sum of
    /// `size^2 * (1 + leaf actions)` over the analysed leaves. A search that
    /// ends earlier never pays for simulation; otherwise it pays at most
    /// about twice the better of the two choices. Both criteria are sound,
    /// so mixing them in one search is too; bounds cached under effective
    /// prices are recomputed after the switch. With
    /// [`Config::reduce_leaves`], the simulation is needed up front, and
    /// this is [`Pruning::EffectiveSimulation`] from the start.
    #[default]
    Adaptive,
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
    /// The dominance criterion. Default: [`Pruning::Adaptive`].
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
    /// Whether to shrink the leaf state spaces by leaf simulation before
    /// searching (Torralba et al. 2016, Section 6, in our own formulation;
    /// their exact rule is in a technical report not checked here).
    /// Default: `false`.
    ///
    /// On every analysed leaf (see [`Config::max_analysis_states`]), the
    /// leaf simulation `⪯` of [`Pruning::Simulation`] is computed once, and
    /// a leaf transition `x -a-> x'` is removed if
    ///
    /// 1. `x' ⪯ x`: staying at `x` is as good, or
    /// 2. another leaf action `b` applicable in `x`, `x -b-> y'`, has
    ///    `cost(b) <= cost(a)`, center preconditions contained in `a`'s, a
    ///    smaller key `(cost, number of center preconditions, index)`, and
    ///    `x' ⪯ y'`.
    ///
    /// Leaf closure then skips removed transitions, so fewer leaf states
    /// are reached and stored ([`Statistics::reduced_leaf_transitions`] and
    /// [`Statistics::unreachable_leaf_states`] report the effect).
    ///
    /// *Soundness.* Take any compliant leaf path `ρ` of the full leaf
    /// system, and its first removed step `s_i -a-> s_{i+1}`. Under rule 2
    /// its replacement `b` may be removed too, but then by a still smaller
    /// key, so the chain ends in a kept step `s_i -c-> z` (or in staying,
    /// rule 1) with `s_{i+1} ⪯ z` by transitivity. `c` is applicable
    /// wherever `a` was (weaker center preconditions) and no dearer. By the
    /// simulation, the rest of `ρ` is mirrored from `z` at no higher cost,
    /// in no more steps, meeting the same global preconditions and ending
    /// in a goal leaf state. Repeating at the next removed step terminates:
    /// each round either fixes one more step or (rule 1) shortens the
    /// path. So every completion of the full system has one of the reduced
    /// system at no higher cost, which preserves optimality and
    /// solvability, and every reduced completion is a real one. The
    /// dominance analyses still use the full transitions, which only makes
    /// frontier and effective-price bounds more conservative, and mirrored
    /// paths in the simulation bound are converted the same way.
    pub reduce_leaves: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pruning: Pruning::default(),
            fold_g: true,
            leaves: Leaves::default(),
            max_analysis_states: 256,
            reduce_leaves: false,
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
    /// Leaf simulations computed (one per analysed leaf, if any criterion or
    /// [`Config::reduce_leaves`] needed them).
    pub simulated_leaves: u64,
    /// Leaf transitions removed by [`Config::reduce_leaves`].
    pub reduced_leaf_transitions: u64,
    /// Leaf states that [`Config::reduce_leaves`] made unreachable from the
    /// initial leaf states (under every center).
    pub unreachable_leaf_states: u64,
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

        let dominance = Dominance::new(&decoupled, self, order == Order::Greedy);
        let mut statistics = Statistics::default();
        if self.reduce_leaves {
            reduce_leaves(&mut decoupled, &dominance, &mut statistics);
        }

        let mut search = Search {
            task: &decoupled,
            heuristic,
            order,
            dominance,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            by_center: BTreeMap::new(),
            best: None,
            statistics,
        };
        search.run();
        search.statistics.simulated_leaves = search.dominance.simulated_leaves();

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

/// Removes the redundant leaf transitions of every leaf with a simulation
/// (see [`Config::reduce_leaves`]) and counts the effect.
fn reduce_leaves(task: &mut DecoupledTask, dominance: &Dominance, statistics: &mut Statistics) {
    let widen = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
    for leaf in 0..task.leaves.len() {
        let Some(rel) = dominance.simulators(leaf) else {
            continue;
        };

        let parts: Vec<_> = task
            .globals
            .iter()
            .filter_map(|g| g.parts[leaf].as_ref())
            .collect();
        let (disabled, lost) = simulation::redundant_transitions(&task.leaves[leaf], &parts, rel);
        statistics.reduced_leaf_transitions += widen(disabled.iter().filter(|&&d| d).count());
        statistics.unreachable_leaf_states += widen(lost);
        task.leaves[leaf].disabled = disabled;
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
        let work = self.task.leaves.iter().map(|l| l.size).sum::<usize>() + 1;
        if self
            .dominance
            .record_work(self.task, u64::try_from(work).unwrap_or(u64::MAX))
        {
            for node in &mut self.nodes {
                node.bounds = None;
            }
        }

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
