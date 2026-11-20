//! Dominance pruning of decoupled states (Torralba, Gnad, Dubbert and
//! Hoffmann 2016), generalised to this crate's global actions and extended
//! with a shared `g` budget.
//!
//! A stored state `t` *dominates* a generated state `n` with the same
//! center if every completion of `n` can be turned into a completion of `t`
//! with the same remaining global actions and no higher total cost. `n` is
//! then pruned. Each [`Pruning`] criterion bounds, for every leaf state `x`
//! reached in `n`, what `t` must pay to do whatever `n` does from `x`: a
//! *bound* `D_t(x)` with the property
//!
//! > (★) for every center path `π` from `t` and every `π`-compliant leaf path
//! > `ρ` from `x`, `t` has a compliant leaf path that does what `ρ` does (ends
//! > in a goal leaf state if `ρ` does, satisfies the same global
//! > preconditions) at total leaf cost at most `D_t(x) + cost(ρ)`.
//!
//! The criteria:
//!
//! - [`Pruning::Basic`]: `D_t(x) = price_t(x)`, on every reached `x`
//!   (Gnad and Hoffmann).
//! - [`Pruning::Frontier`]: the same bound, compared only on the frontier
//!   of `n` (Definition 2 of the paper): leaf states that are *anchored* or
//!   that some leaf action leaves at a strictly cheaper price than its
//!   target's. If `ρ` starts outside the frontier, its first state on the
//!   frontier is no dearer in `n` than `x` plus the cost to get there.
//! - [`Pruning::EffectivePrices`]: `D_t` is the effective price
//!   (Definition 4): the point-wise least `p` with `p(x) = price_t(x)` on
//!   anchored states and `p(x) = min(price_t(x), max_{x -a-> y} p(y) - cost(a))`
//!   elsewhere, over leaf actions regardless of their center preconditions.
//! - [`Pruning::Simulation`]: `D_t(x) = min_{x ⪯ y} price_t(y)` for the leaf
//!   simulation `⪯` of [`simulation`] (Definition 6).
//! - [`Pruning::EffectiveSimulation`]: effective prices over simulation
//!   bounds (Definition 7), subsuming all others.
//!
//! *Anchored* leaf states generalise the paper's single goal leaf state to
//! global actions that touch leaves: a leaf state is anchored if it
//! satisfies the leaf goal or the leaf preconditions of a global action's
//! part on the leaf (every state is anchored if some part has none). An
//! unanchored leaf state can only be left by leaf actions, which is what the
//! frontier and effective-price arguments need; the anchored ones are where
//! the leaf path meets the goal or a global action, so their bound must be a
//! real price.
//!
//! *Proof of (★) for effective prices.* Let `ρ` visit `x = s0, ..., sm`. Take
//! the first `i` with `p(s_i) = S(s_i)`, where `S` is `price_t` (or the
//! simulation bound); it exists because `ρ` ends in a goal leaf state, which
//! is anchored. Before `i` all states are unanchored, so every step is a
//! leaf action (a global action touching the leaf would anchor it) and
//! `p(s_j) >= p(s_{j+1}) - cost(a_{j+1})`; summing,
//! `p(s_i) <= p(x) + cost(s0..s_i)`. `t` reaches `s_i` (or a simulating
//! state `y`, then mirrors the rest of `ρ` from `y` by the simulation) at
//! price `S(s_i) = p(s_i)` and follows the rest of `ρ`. States whose least
//! value stays at `-∞` cannot reach an anchored state at all, so no
//! completion uses them. The frontier and plain cases are special cases
//! with `p = price_t` and simulation bounds with `p = S`.
//!
//! *The `g` budget.* Leaf prices exclude `g`. For each leaf let
//! `δ_L = max_x (D_t(x) - price_n(x))` over the compared states `x`. A
//! completion of `n` through leaf states `x_L` costs
//! `g(n) + Σ_L (price_n(x_L) + rest_L) + global rest`, and by (★) one of `t`
//! costs at most `g(t) + Σ_L (price_n(x_L) + δ_L + rest_L) + global rest`.
//! So `t` dominates `n` for cost-optimal search if
//! `g(t) + Σ_L δ_L <= g(n)`, with `fold_g`; without it, the classic
//! condition `g(t) <= g(n)` and `δ_L <= 0` for every leaf is used. Comparing
//! `g + price` per leaf state instead would be unsound with two or more
//! leaves: `g` is paid once but would be credited per leaf (the
//! `slack_is_shared_between_leaves` test is a counterexample). For greedy
//! search only reachability matters: `t` dominates `n` if every compared
//! bound is finite.
//!
//! *Optimality of A* with this pruning.* Say a stored node *covers* cost `C`
//! via global actions `π` if applying `π` to it yields a goal state with
//! total cost at most `C`. While no solution of cost `C*` is recorded, take
//! a covering stored node with shortest `π`. If `π` is empty, the node is a
//! goal and its cost `<= C*` was recorded when it was stored. Otherwise, if
//! it were closed, its successor along `π[0]` was either stored (covering
//! via a shorter path) or pruned by a stored node that covers via the same,
//! shorter `π[1..]` (dominance keeps the remaining global actions), or
//! declared a dead end, which an admissible heuristic only does without
//! completions. So the node is open, with `f <= C*`, and the search does
//! not stop before recording `C*`.
//!
//! Leaves whose transitions read the center (center-conditioned effects,
//! derived slots) or that have more than
//! [`Config::max_analysis_states`](crate::Config::max_analysis_states)
//! states always use the basic bound.

mod simulation;

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;

use crate::decoupled::{DecoupledState, DecoupledTask, LeafPart, Price, Prices};
use crate::{Config, Pruning};
use simulation::Bits;

/// A bound value: a price, `+∞` for "not reachable", `-∞` for "no purpose".
type Bound = i128;

/// The bound of a leaf state `t` cannot reproduce.
const INFINITE: Bound = Bound::MAX;

/// The bound of a leaf state no completion can use.
const NO_PURPOSE: Bound = Bound::MIN;

/// What the dominance check knows about one leaf beyond its prices.
#[derive(Debug)]
struct Analysis {
    /// Whether each leaf state is anchored (goal or global precondition).
    anchored: Vec<bool>,
    /// Per leaf state: `(successor, cost)` for every leaf action applicable
    /// under some center.
    successors: Vec<Vec<(usize, Price)>>,
    /// Per leaf state: `(predecessor, cost)`, the reverse of `successors`.
    predecessors: Vec<Vec<(usize, Price)>>,
    /// Per leaf state, its simulators; `None` unless the criterion uses
    /// simulation.
    simulators: Option<Vec<Bits>>,
}

/// The dominance criterion compiled for one decoupled task.
#[derive(Debug)]
pub(crate) struct Dominance {
    /// The criterion.
    pruning: Pruning,
    /// Whether `g` differences may be paid for by cheaper prices.
    fold_g: bool,
    /// Whether only reachability matters (greedy search).
    greedy: bool,
    /// Per leaf, its analysis; `None` for leaves that use the basic bound.
    leaves: Vec<Option<Analysis>>,
}

/// The bounds `D_t` of a stored state, per leaf; `None` for leaves where
/// the bound is the price itself.
pub(crate) type Bounds = Vec<Option<Vec<Bound>>>;

/// The frontier of a generated state, per leaf; `None` for leaves compared
/// on every reached state.
pub(crate) type Frontier = Vec<Option<Vec<bool>>>;

impl Dominance {
    /// Compiles the criterion of `config` for `task`; `greedy` drops cost
    /// comparisons.
    pub(crate) fn new(task: &DecoupledTask, config: &Config, greedy: bool) -> Self {
        let analysed = config.pruning != Pruning::Basic;
        let leaves = (0..task.leaves.len())
            .map(|leaf| {
                let parts: Vec<&LeafPart> = task
                    .globals
                    .iter()
                    .filter_map(|g| g.parts[leaf].as_ref())
                    .collect();
                let compiled = &task.leaves[leaf];
                (analysed
                    && compiled.size <= config.max_analysis_states
                    && !compiled.reads_center(parts.iter().copied()))
                .then(|| analyse(task, leaf, &parts, config.pruning))
            })
            .collect();

        Self {
            pruning: config.pruning,
            fold_g: config.fold_g,
            greedy,
            leaves,
        }
    }

    /// Whether the criterion compares only on frontiers.
    fn uses_frontier(&self) -> bool {
        self.pruning == Pruning::Frontier
    }

    /// The bounds of `state` as a dominating state.
    pub(crate) fn bounds(&self, task: &DecoupledTask, state: &DecoupledState) -> Bounds {
        self.leaves
            .iter()
            .zip(&state.prices)
            .enumerate()
            .map(|(leaf, (analysis, prices))| {
                let analysis = analysis.as_ref()?;
                (!self.uses_frontier()).then(|| {
                    let size = task.leaves[leaf].size;
                    self.leaf_bounds(analysis, prices, size)
                })
            })
            .collect()
    }

    /// The bounds of one leaf with `prices` under `analysis`.
    fn leaf_bounds(&self, analysis: &Analysis, prices: &Prices, size: usize) -> Vec<Bound> {
        let base = if let Some(simulators) = &analysis.simulators {
            simulation_bounds(simulators, prices, size)
        } else {
            let mut base = vec![INFINITE; size];
            for (rank, price) in prices.iter() {
                base[rank] = Bound::from(price);
            }

            base
        };

        match self.pruning {
            Pruning::EffectivePrices | Pruning::EffectiveSimulation => {
                effective_prices(analysis, &base)
            }
            Pruning::Basic | Pruning::Frontier | Pruning::Simulation => base,
        }
    }

    /// The frontier of `state` as a dominated state.
    pub(crate) fn frontier(&self, task: &DecoupledTask, state: &DecoupledState) -> Frontier {
        self.leaves
            .iter()
            .zip(&state.prices)
            .enumerate()
            .map(|(leaf, (analysis, prices))| {
                let analysis = analysis.as_ref().filter(|_| self.uses_frontier())?;
                let dense = prices.to_dense(task.leaves[leaf].size);
                Some(
                    (0..dense.len())
                        .map(|x| is_frontier(analysis, &dense, x))
                        .collect(),
                )
            })
            .collect()
    }

    /// Whether stored state `t` (reached at `g_t`, with bounds `bounds_t`)
    /// dominates generated state `n` (reached at `g_n`, with frontier
    /// `frontier_n`). The centers must be equal.
    pub(crate) fn dominates(
        &self,
        (t, g_t, bounds_t): (&DecoupledState, Price, &Bounds),
        (n, g_n, frontier_n): (&DecoupledState, Price, &Frontier),
    ) -> bool {
        let mut budget: Bound = 0;
        for (leaf, (prices_n, prices_t)) in n.prices.iter().zip(&t.prices).enumerate() {
            let Some(delta) = slack(
                prices_n,
                prices_t,
                bounds_t[leaf].as_deref(),
                frontier_n[leaf].as_deref(),
            ) else {
                return false;
            };

            if !self.fold_g && delta > 0 {
                return false;
            }

            budget = budget.saturating_add(delta);
        }

        if self.greedy {
            return true;
        }

        let (g_t, g_n) = (Bound::from(g_t), Bound::from(g_n));
        if self.fold_g {
            g_t.saturating_add(budget) <= g_n
        } else {
            g_t <= g_n
        }
    }
}

/// `δ_L = max (D_t(x) - price_n(x))` over the reached leaf states `x` of `n`
/// (on the frontier only, if given); `None` if some bound is infinite, and
/// [`NO_PURPOSE`] if nothing is compared.
fn slack(
    prices_n: &Prices,
    prices_t: &Prices,
    bounds_t: Option<&[Bound]>,
    frontier_n: Option<&[bool]>,
) -> Option<Bound> {
    let mut delta = NO_PURPOSE;
    let mut t = prices_t.iter().peekable();
    for (rank, price) in prices_n.iter() {
        if frontier_n.is_some_and(|f| !f[rank]) {
            continue;
        }

        let bound = if let Some(bounds) = bounds_t {
            bounds[rank]
        } else {
            while t.next_if(|&(r, _)| r < rank).is_some() {}
            t.next_if(|&(r, _)| r == rank)
                .map_or(INFINITE, |(_, p)| Bound::from(p))
        };
        if bound == INFINITE {
            return None;
        }

        delta = delta.max(bound.saturating_sub(Bound::from(price)));
    }

    Some(delta)
}

/// The static analysis of leaf `leaf` for criterion `pruning`.
fn analyse(task: &DecoupledTask, leaf: usize, parts: &[&LeafPart], pruning: Pruning) -> Analysis {
    let compiled = &task.leaves[leaf];
    let view = compiled.static_view();
    let anchored = (0..compiled.size)
        .map(|x| view.is_goal(x) || parts.iter().any(|p| view.satisfies(x, &p.pre)))
        .collect();

    let mut successors = vec![Vec::new(); compiled.size];
    let mut predecessors = vec![Vec::new(); compiled.size];
    for (x, out) in successors.iter_mut().enumerate() {
        for a in compiled
            .actions
            .iter()
            .filter(|a| view.satisfies(x, &a.part.pre))
        {
            let y = view.apply(x, &a.part.eff);
            out.push((y, a.cost));
            predecessors[y].push((x, a.cost));
        }
    }

    let simulators = matches!(pruning, Pruning::Simulation | Pruning::EffectiveSimulation)
        .then(|| simulation::simulators(compiled, parts));

    Analysis {
        anchored,
        successors,
        predecessors,
        simulators,
    }
}

/// Whether leaf state `x` is on the frontier of the dense prices `prices`:
/// anchored, or left by some leaf action at a strictly lower price than its
/// target has.
fn is_frontier(analysis: &Analysis, prices: &[Price], x: usize) -> bool {
    analysis.anchored[x]
        || analysis.successors[x]
            .iter()
            .any(|&(y, cost)| prices[x].saturating_add(cost) < prices[y])
}

/// `S(x) = min over reached y with x ⪯ y of price(y)`, [`INFINITE`] if none.
fn simulation_bounds(simulators: &[Bits], prices: &Prices, size: usize) -> Vec<Bound> {
    (0..size)
        .map(|x| {
            prices
                .iter()
                .filter(|&(y, _)| simulators[x].contains(y))
                .map(|(_, p)| Bound::from(p))
                .min()
                .unwrap_or(INFINITE)
        })
        .collect()
}

/// The point-wise least `p` with `p(x) = base(x)` on anchored states and
/// `p(x) = min(base(x), max_{x -a-> y} p(y) - cost(a))` elsewhere.
///
/// Dijkstra with a max-heap from the anchored states backwards: values only
/// fall along an edge (`p(y) - cost <= p(y)`), so a popped state is final.
/// States never reached backwards keep [`NO_PURPOSE`].
fn effective_prices(analysis: &Analysis, base: &[Bound]) -> Vec<Bound> {
    let mut p: Vec<Bound> = base
        .iter()
        .zip(&analysis.anchored)
        .map(|(&b, &anchored)| if anchored { b } else { NO_PURPOSE })
        .collect();
    let mut settled = vec![false; p.len()];
    let mut open: BinaryHeap<(Bound, usize)> = p
        .iter()
        .enumerate()
        .filter(|&(x, _)| analysis.anchored[x])
        .map(|(x, &v)| (v, x))
        .collect();

    while let Some((value, y)) = open.pop() {
        if core::mem::replace(&mut settled[y], true) {
            continue;
        }

        for &(x, cost) in &analysis.predecessors[y] {
            if settled[x] || analysis.anchored[x] {
                continue;
            }

            let through = if value == INFINITE {
                INFINITE
            } else {
                value.saturating_sub(Bound::from(cost))
            };
            let candidate = base[x].min(through);
            if candidate > p[x] {
                p[x] = candidate;
                open.push((candidate, x));
            }
        }
    }

    p
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

    use super::*;
    use crate::Factoring;
    use crate::decoupled::UNREACHED;

    /// A center `c` and two binary leaves `x` and `y`, each set by a leaf
    /// action of cost 1.
    fn two_leaves() -> DecoupledTask {
        let set = |var| Operator {
            name: "set".into(),
            pre: vec![],
            eff: vec![Effect::new(Fact::new(var, 1))],
            cost: 1,
        };
        let task = Task {
            variables: (0..3)
                .map(|v| Variable::new(alloc::format!("v{v}"), 2))
                .collect(),
            operators: vec![set(1), set(2)],
            initial: State(vec![0, 0, 0]),
            goal: vec![Fact::new(1, 1), Fact::new(2, 1)],
            axioms: vec![],
        };
        let factoring = Factoring::new(&task, vec![vec![1], vec![2]]).expect("valid");
        DecoupledTask::new(&task, &factoring).expect("supported")
    }

    /// A state with center 0 and leaf state 0 of both leaves at `price`.
    fn state(price: Price) -> DecoupledState {
        let leaf = Prices::from_dense(&[price, UNREACHED]);
        DecoupledState {
            center: vec![0],
            prices: vec![leaf.clone(), leaf],
        }
    }

    /// Whether `(t, g_t)` dominates `(n, g_n)` under basic bounds.
    fn dominates(fold_g: bool, (t, g_t): (Price, Price), (n, g_n): (Price, Price)) -> bool {
        let task = two_leaves();
        let config = Config {
            pruning: Pruning::Basic,
            fold_g,
            ..Config::default()
        };
        let dominance = Dominance::new(&task, &config, false);
        let (t, n) = (state(t), state(n));
        let bounds = dominance.bounds(&task, &t);
        let frontier = dominance.frontier(&task, &n);
        dominance.dominates((&t, g_t, &bounds), (&n, g_n, &frontier))
    }

    /// `t` (g 0, both leaves at 10) against `n` (g 10, both at 0): per leaf
    /// `g + price` ties, but completing both leaves costs 20 from `t` and
    /// 10 from `n`, so the budget must be shared, not granted per leaf.
    #[test]
    fn slack_is_shared_between_leaves() {
        assert!(!dominates(true, (10, 0), (0, 10)));
        assert!(!dominates(false, (10, 0), (0, 10)));
    }

    /// With prices 5 against 0, the shared budget of 10 covers both leaves,
    /// which the plain `g` comparison cannot see.
    #[test]
    fn folded_g_pays_for_dearer_prices() {
        assert!(dominates(true, (5, 0), (0, 10)));
        assert!(!dominates(false, (5, 0), (0, 10)));
        assert!(dominates(false, (0, 0), (5, 10)));
        assert!(!dominates(true, (6, 0), (0, 11)), "12 > 11");
    }
}
