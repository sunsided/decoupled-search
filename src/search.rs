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
use alloc::vec;
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
    /// Anytime decoupled A*: by `g` plus a root heuristic (remaining global
    /// cost only), pruning by `g + h` against the best solution, stopping
    /// once the popped key plus the least possible goal price reaches it.
    Anytime,
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
    /// searching: the superfluous-transition rule of Torralba, Gnad,
    /// Dubbert and Hoffmann (IJCAI 2016, Section 6 of the extended
    /// technical report), with a removal order that makes it sound.
    /// Default: `false`.
    ///
    /// On every analysed leaf (see [`Config::max_analysis_states`]), the
    /// leaf simulation `⪯` of [`Pruning::Simulation`] is computed once.
    /// Say that `x -b-> y'` *replaces* `x -a-> x'` if `cost(b) <= cost(a)`,
    /// `b`'s center preconditions are contained in `a`'s, and `x' ⪯ y'`. A
    /// leaf transition `x -a-> x'` is removed if
    ///
    /// 1. `x' ⪯ x`: staying at `x` is as good, or
    /// 2. another leaf action `b` applicable in `x` replaces it, and either
    ///    `a` does not replace `b` in turn or `b` has the smaller index.
    ///
    /// The report states rule 2 without the last condition and without a
    /// removal order; read as one simultaneous step, it would remove both
    /// of two transitions that replace each other (for example two copies
    /// of one action), which is unsound. Removing one transition at a time,
    /// each justified by one still present, removes exactly what the rules
    /// above remove, up to which of two mutually replacing transitions
    /// stays. The report also drops leaf states that become unreachable and
    /// leaf actions without transitions; here leaf closure skips removed
    /// transitions, so those states are never reached and stored
    /// ([`Statistics::reduced_leaf_transitions`] and
    /// [`Statistics::unreachable_leaf_states`] report the effect).
    /// [`Config::reduced_task`] drops the operators left without a useful
    /// transition, for building heuristics.
    ///
    /// *Soundness* (the report gives no proof). "Replaces" is reflexive and
    /// transitive on the transitions leaving `x`; ordering by it, and by
    /// index between transitions that replace each other, is a strict
    /// partial order on a finite set, and rule 2 removes only transitions
    /// below a greater one. Take any compliant leaf path `ρ` of the full
    /// leaf system, and its first removed step `s_i -a-> s_{i+1}`.
    /// Following rule 2 upwards ends in a maximal transition `s_i -c-> z`,
    /// which rule 2 keeps, with `s_{i+1} ⪯ z` by transitivity; if rule 1
    /// removes it, staying at `s_i` (`z ⪯ s_i`) is as good. `c` is
    /// applicable wherever `a` was (weaker center preconditions) and no
    /// dearer. By the simulation, the rest of `ρ` is mirrored from `z` (or
    /// `s_i`) at no higher cost, in no more steps, meeting the same global
    /// preconditions and ending in a goal leaf state. Repeating at the next
    /// removed step terminates: each round either fixes one more step or
    /// (staying) shortens the path. So every completion of the full system
    /// has one of the reduced system at no higher cost, which preserves
    /// optimality and solvability, and every reduced completion is a real
    /// one. The dominance analyses still use the full transitions, which
    /// only makes frontier and effective-price bounds more conservative,
    /// and mirrored paths in the simulation bound are converted the same
    /// way.
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
    /// Generated states pruned by anytime A* ([`Config::anytime_astar`])
    /// because `g` plus the fork heuristic reaches the best solution cost.
    pub bound_pruned: u64,
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
    /// Under [`Order::Anytime`], the fork heuristic that prunes against the
    /// best solution.
    heuristic: Option<&'t dyn DecoupledHeuristic>,
    /// The root heuristic of [`Order::Anytime`], evaluated with every
    /// reached fact at price 0; `None` otherwise.
    root: Option<&'t dyn DecoupledHeuristic>,
    /// A lower bound on the goal price of every decoupled goal state (see
    /// [`min_goal_price`]); 0 unless [`Order::Anytime`].
    min_goal_price: Price,
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
    /// The most expansions to make; `None` for no limit.
    limit: Option<u64>,
    /// What a sample search measures.
    probe: Probe,
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
        self.run(task, factoring, (None, None), Order::Cost)
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
        self.run(task, factoring, (Some(&heuristic), None), Order::Cost)
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
        self.run(task, factoring, (Some(&heuristic), None), Order::Greedy)
    }

    /// Anytime decoupled A* (Gnad and Hoffmann's Anytime Fork-Root A*,
    /// ICAPS 2015, generalised to star factorings): A* ordered by `g` plus
    /// the `root` heuristic, which estimates only the cost of the remaining
    /// global actions, with `fork` (an estimate of the whole remaining cost,
    /// as for [`Config::astar`]) used to prune nodes that cannot beat the
    /// best solution found so far.
    ///
    /// The root heuristic sees every reached leaf fact at price 0. Build it
    /// on [`Factoring::with_free_leaf_actions`], for example
    /// `HMax::new(&factoring.with_free_leaf_actions(&task)?)`: any
    /// completion of a member state is then a plan of that task costing
    /// only its global actions, so `h^max` there is root-admissible.
    ///
    /// The search records solutions when they are generated and stops once
    /// the popped key plus the least goal price any decoupled goal state can
    /// have (per leaf, the cheapest leaf-action cost from the initial leaf
    /// state to a goal leaf state, global parts free, ignoring the center)
    /// reaches the best solution cost, or when the open list is empty. This
    /// subsumes the paper's early termination (an expanded goal whose goal
    /// price equals that minimum) and is sound by the same argument as
    /// [`Config::astar`]: while the optimum `C*` is not recorded, a node
    /// covering it is open, with key at most `C*` minus its goal price.
    /// Nodes with `g + fork` at least the best cost are pruned
    /// ([`Statistics::bound_pruned`]), which cannot cut below `C*`.
    ///
    /// The plan is cost-optimal if `root` is root-admissible and `fork` is
    /// admissible.
    ///
    /// # Errors
    ///
    /// As [`Config::search`].
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Config, Factoring, HMax};
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
    ///         op(vec![Fact::new(0, 0), Fact::new(1, 0)], Fact::new(1, 2)),
    ///         op(vec![Fact::new(0, 1), Fact::new(1, 2)], Fact::new(1, 1)),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let factoring = Factoring::fork(&task);
    /// let root = HMax::new(&factoring.with_free_leaf_actions(&task)?);
    /// let outcome = Config::default().anytime_astar(&task, &factoring, &root, &HMax::new(&task))?;
    /// assert_eq!(outcome.plan.map(|p| p.cost), Some(3));
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    pub fn anytime_astar(
        &self,
        task: &Task,
        factoring: &Factoring,
        root: &(impl DecoupledHeuristic + ?Sized),
        fork: &(impl DecoupledHeuristic + ?Sized),
    ) -> Result<Outcome, Error> {
        self.run(task, factoring, (Some(&fork), Some(&root)), Order::Anytime)
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
        (heuristic, root): (
            Option<&dyn DecoupledHeuristic>,
            Option<&dyn DecoupledHeuristic>,
        ),
        order: Order,
    ) -> Result<Outcome, Error> {
        self.probe(task, factoring, (heuristic, root), order, None)
            .map(|(outcome, _)| outcome)
    }

    /// [`Config::run`] that stops after `limit` expansions, if given, and
    /// also reports what a sample search measures.
    ///
    /// # Errors
    ///
    /// As [`Config::search`].
    fn probe(
        &self,
        task: &Task,
        factoring: &Factoring,
        (heuristic, root): (
            Option<&dyn DecoupledHeuristic>,
            Option<&dyn DecoupledHeuristic>,
        ),
        order: Order,
        limit: Option<u64>,
    ) -> Result<(Outcome, Probe), Error> {
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

        let min_goal_price = if order == Order::Anytime {
            min_goal_price(&decoupled)
        } else {
            0
        };
        let mut search = Search {
            task: &decoupled,
            heuristic,
            root,
            min_goal_price,
            order,
            dominance,
            nodes: Vec::new(),
            open: BinaryHeap::new(),
            by_center: BTreeMap::new(),
            best: None,
            statistics,
            limit,
            probe: Probe::default(),
        };
        search.run();
        search.statistics.simulated_leaves = search.dominance.simulated_leaves();

        let plan = search.best.map(|(cost, id)| {
            let plan = reconstruct::plan(&decoupled, &search.path(id), cost);
            with_true_cost(task, plan, decoupled.is_reachability_only())
        });

        let outcome = Outcome {
            plan,
            statistics: search.statistics,
        };
        Ok((outcome, search.probe))
    }
}

/// What a sample search measures (see [`sample`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Probe {
    /// Whether the search stopped at its expansion limit rather than by
    /// finishing.
    pub(crate) cut_off: bool,
    /// Work units spent: per generated decoupled state, one plus the number
    /// of leaf states its leaf closures range over.
    pub(crate) work: u64,
    /// The heuristic value of the initial state; `None` if it was pruned.
    pub(crate) initial_h: Option<Price>,
    /// The least heuristic value of any stored state.
    pub(crate) best_h: Option<Price>,
}

/// A sample search for factoring selection: decoupled A* with `h^max`
/// (cost-optimal) or greedy best-first search with `h^FF` (not), each with
/// the default [`Config`], stopped after `expansions` expansions. Returns
/// the probe and the number of expansions.
///
/// # Errors
///
/// As [`Config::search`].
pub(crate) fn sample(
    task: &Task,
    factoring: &Factoring,
    optimal: bool,
    expansions: u64,
) -> Result<(Probe, u64), Error> {
    let config = Config::default();
    let (outcome, probe) = if optimal {
        let h = crate::HMax::new(task);
        config.probe(
            task,
            factoring,
            (Some(&h), None),
            Order::Cost,
            Some(expansions),
        )?
    } else {
        let h = crate::HFf::new(task);
        config.probe(
            task,
            factoring,
            (Some(&h), None),
            Order::Greedy,
            Some(expansions),
        )?
    };

    Ok((probe, outcome.statistics.expanded))
}

impl Config {
    /// `task` without the operators that [`Config::reduce_leaves`] makes
    /// useless under `factoring`, for building heuristics (the last step of
    /// the leaf reduction in Torralba, Gnad, Dubbert and Hoffmann's
    /// technical report).
    ///
    /// An operator is removed if it is a leaf action of an analysed leaf
    /// (see [`Config::max_analysis_states`]) and none of its kept leaf
    /// transitions starts in a leaf state that the reduced leaf can still
    /// reach (over kept leaf actions and global parts, under any center).
    /// Variables, goal and the remaining operators are unchanged, so fact
    /// prices of the search fit heuristics built on the result.
    ///
    /// *Soundness.* Only use heuristics built on the result in searches
    /// with `reduce_leaves: true`, the same `factoring` and the same
    /// `max_analysis_states`. There, every reached leaf state is reachable
    /// in the reduced leaf, and by the soundness argument on
    /// [`Config::reduce_leaves`] every decoupled state has a cheapest
    /// completion that uses only kept transitions from such states, hence
    /// only operators of the result. A heuristic admissible on the result
    /// is therefore admissible for the search, and fewer operators can only
    /// raise its estimates. Without leaf reduction in the search the
    /// estimates may exceed the true cost.
    ///
    /// The result is the same for both [`Leaves`] modes: the simulation
    /// and the removed transitions do not depend on it.
    ///
    /// # Errors
    ///
    /// As [`Config::search`], except that [`Config::leaves`] is ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Config, Factoring, HMax, Pruning};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A truck over three locations and a package (3: in the truck) that
    /// // must go from location 0 to location 1.
    /// let op = |pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let mut operators = Vec::new();
    /// for x in 0..3 {
    ///     for y in (0..3).filter(|&y| y != x) {
    ///         operators.push(op(vec![Fact::new(0, x)], Fact::new(0, y)));
    ///     }
    ///     operators.push(op(vec![Fact::new(0, x), Fact::new(1, x)], Fact::new(1, 3)));
    ///     operators.push(op(vec![Fact::new(0, x), Fact::new(1, 3)], Fact::new(1, x)));
    /// }
    /// let task = Task {
    ///     variables: vec![Variable::new("truck", 3), Variable::new("pkg", 4)],
    ///     operators,
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let config = Config { reduce_leaves: true, ..Config::default() };
    /// let factoring = Factoring::fork(&task);
    /// let reduced = config.reduced_task(&task, &factoring)?;
    /// // Only loading at 0 and unloading at 1 remain of the package's six.
    /// assert_eq!(reduced.operators.len(), task.operators.len() - 4);
    ///
    /// let outcome = config.astar(&task, &factoring, &HMax::new(&reduced))?;
    /// assert_eq!(outcome.plan.map(|p| p.cost), Some(3));
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    pub fn reduced_task(&self, task: &Task, factoring: &Factoring) -> Result<Task, Error> {
        let mut decoupled = DecoupledTask::new(task, factoring)?;
        let config = Config {
            reduce_leaves: true,
            ..*self
        };
        let dominance = Dominance::new(&decoupled, &config, false);
        reduce_leaves(&mut decoupled, &dominance, &mut Statistics::default());

        let mut removed = vec![false; task.operators.len()];
        for (index, leaf) in decoupled.leaves.iter().enumerate() {
            if dominance.simulators(index).is_none() {
                continue;
            }

            let parts: Vec<_> = decoupled
                .globals
                .iter()
                .filter_map(|g| g.parts[index].as_ref())
                .collect();
            let live = simulation::live_actions(leaf, &parts);
            for (action, live) in leaf.actions.iter().zip(live) {
                removed[action.op] |= !live;
            }
        }

        let mut reduced = task.clone();
        let mut keep = removed.iter().map(|&r| !r);
        reduced.operators.retain(|_| keep.next().unwrap_or(true));
        Ok(reduced)
    }
}

/// A lower bound on the goal price of every decoupled goal state of `task`:
/// per leaf, the cheapest leaf-action cost from the initial leaf state to a
/// goal leaf state, global parts free, ignoring the center (0 for leaves
/// whose transitions read the center); [`Price::MAX`] if some leaf can
/// never reach its goal.
fn min_goal_price(task: &DecoupledTask) -> Price {
    task.leaves
        .iter()
        .enumerate()
        .map(|(index, leaf)| {
            let parts: Vec<_> = task
                .globals
                .iter()
                .filter_map(|g| g.parts[index].as_ref())
                .collect();
            if leaf.reads_center(parts.iter().copied()) {
                return 0;
            }

            simulation::cheapest_goal(leaf, &parts).unwrap_or(Price::MAX)
        })
        .fold(0, Price::saturating_add)
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

            if self
                .limit
                .is_some_and(|limit| self.statistics.expanded >= limit)
            {
                self.probe.cut_off = true;
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
            Order::Anytime => self
                .best
                .is_some_and(|(cost, _)| key.saturating_add(self.min_goal_price) >= cost),
            Order::Greedy => self.best.is_some(),
        }
    }

    /// The heuristic estimate of `state`; 0 without a heuristic, `None` for
    /// a dead end.
    fn estimate(&self, state: &DecoupledState) -> Option<Price> {
        self.heuristic
            .map_or(Some(0), |h| h.estimate(&self.task.fact_prices(state)))
    }

    /// The open-list key of `state` reached at `g` with heuristic value `h`;
    /// `None` if the root heuristic of [`Order::Anytime`] declares it a dead
    /// end.
    fn key(&self, state: &DecoupledState, g: Price, h: Price) -> Option<Price> {
        match self.order {
            Order::Cost => Some(g.saturating_add(h)),
            Order::Greedy => Some(h),
            Order::Anytime => {
                let root = self.root.map_or(Some(0), |r| {
                    r.estimate(&self.task.fact_prices(state).reached_for_free())
                })?;
                Some(g.saturating_add(root))
            }
        }
    }

    /// Adds `state` reached at cost `g` unless a stored node dominates it or
    /// the heuristic declares it a dead end, and records it as the best
    /// solution if it is one.
    fn insert(&mut self, state: DecoupledState, g: Price, parent: Option<(usize, usize)>) {
        self.statistics.generated += 1;
        let work = self.task.leaves.iter().map(|l| l.size).sum::<usize>() + 1;
        let work = u64::try_from(work).unwrap_or(u64::MAX);
        self.probe.work = self.probe.work.saturating_add(work);
        if self.dominance.record_work(self.task, work) {
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

        self.probe.initial_h.get_or_insert(h);
        let Some(key) = self.key(&state, g, h) else {
            self.statistics.dead_ends += 1;
            return;
        };

        if self.order == Order::Anytime
            && self
                .best
                .is_some_and(|(best, _)| g.saturating_add(h) >= best)
        {
            self.statistics.bound_pruned += 1;
            return;
        }

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
        self.probe.best_h = Some(self.probe.best_h.map_or(h, |b| b.min(h)));
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
