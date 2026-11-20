//! Dominance pruning of decoupled states: the criteria of [`Pruning`],
//! whose documentation states and proves them, and the per-leaf analyses
//! they need (anchored states, leaf transitions, [`simulation`]).
//!
//! Leaves whose transitions read the center (center-conditioned effects,
//! derived slots) or that have more than
//! [`Config::max_analysis_states`](crate::Config::max_analysis_states)
//! states always use the basic bound.

pub(crate) mod simulation;

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;

use crate::decoupled::{DecoupledState, DecoupledTask, LeafPart, Price, Prices};
use crate::{Config, Pruning};
pub(crate) use simulation::Bits;

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
}

/// When [`Pruning::Adaptive`] switches from effective prices to effective
/// simulation.
#[derive(Debug)]
struct Escalation {
    /// Estimated cost of computing every leaf simulation, in the units of
    /// [`Dominance::record_work`].
    budget: u64,
    /// Work recorded so far.
    spent: u64,
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
    /// Per leaf, its simulators, once computed; `None` for leaves without
    /// (center-dependent, too large, or not needed).
    simulations: Vec<Option<Vec<Bits>>>,
    /// Per leaf, whether it may be analysed: small enough and with
    /// center-independent transitions.
    eligible: Vec<bool>,
    /// The pending switch of [`Pruning::Adaptive`], if any.
    escalation: Option<Escalation>,
}

/// The bounds `D_t` of a stored state, per leaf; `None` for leaves where
/// the bound is the price itself.
pub(crate) type Bounds = Vec<Option<Vec<Bound>>>;

/// The frontier of a generated state, per leaf; `None` for leaves compared
/// on every reached state.
pub(crate) type Frontier = Vec<Option<Vec<bool>>>;

impl Dominance {
    /// Compiles the criterion of `config` for `task`; `greedy` drops cost
    /// comparisons. Computes leaf simulations up front if the criterion
    /// uses them or [`Config::reduce_leaves`] is set; [`Pruning::Adaptive`]
    /// otherwise starts with effective prices.
    pub(crate) fn new(task: &DecoupledTask, config: &Config, greedy: bool) -> Self {
        let eligible: Vec<bool> = (0..task.leaves.len())
            .map(|leaf| {
                let compiled = &task.leaves[leaf];
                compiled.size <= config.max_analysis_states
                    && !compiled.reads_center(parts(task, leaf))
            })
            .collect();
        let pruning = match config.pruning {
            Pruning::Adaptive if config.reduce_leaves => Pruning::EffectiveSimulation,
            Pruning::Adaptive => Pruning::EffectivePrices,
            other => other,
        };
        let leaves = eligible
            .iter()
            .enumerate()
            .map(|(leaf, &ok)| (ok && pruning != Pruning::Basic).then(|| analyse(task, leaf)))
            .collect();

        let mut dominance = Self {
            pruning,
            fold_g: config.fold_g,
            greedy,
            leaves,
            simulations: vec![None; task.leaves.len()],
            eligible,
            escalation: None,
        };
        if uses_simulation(pruning) || config.reduce_leaves {
            dominance.simulate(task);
        } else if config.pruning == Pruning::Adaptive {
            let budget = dominance.simulation_cost(task);
            dominance.escalation = (budget > 0).then_some(Escalation { budget, spent: 0 });
        }

        dominance
    }

    /// Computes the simulation of every eligible leaf.
    fn simulate(&mut self, task: &DecoupledTask) {
        for (leaf, simulation) in self.simulations.iter_mut().enumerate() {
            if self.eligible[leaf] && simulation.is_none() {
                let parts: Vec<&LeafPart> = parts(task, leaf).collect();
                *simulation = Some(simulation::simulators(&task.leaves[leaf], &parts));
            }
        }
    }

    /// An estimate of what [`Dominance::simulate`] costs: per eligible leaf,
    /// `size^2` pairs times one plus its number of leaf actions.
    fn simulation_cost(&self, task: &DecoupledTask) -> u64 {
        let widen = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        task.leaves
            .iter()
            .zip(&self.eligible)
            .filter(|&(_, &ok)| ok)
            .map(|(leaf, _)| {
                let size = widen(leaf.size);
                size.saturating_mul(size)
                    .saturating_mul(widen(leaf.actions.len()) + 1)
            })
            .fold(0, u64::saturating_add)
    }

    /// Records `work` units of search effort (the leaf states of one
    /// generated decoupled state). Under [`Pruning::Adaptive`], once the
    /// search has spent as much as the leaf simulations are estimated to
    /// cost, computes them and switches to effective simulation; returns
    /// `true` then, so that cached bounds are recomputed.
    ///
    /// This is the ski-rental rule: on tasks the search finishes cheaply
    /// the simulation is never paid for, and otherwise it costs at most
    /// about as much as the search has already spent.
    pub(crate) fn record_work(&mut self, task: &DecoupledTask, work: u64) -> bool {
        let Some(escalation) = &mut self.escalation else {
            return false;
        };

        escalation.spent = escalation.spent.saturating_add(work);
        if escalation.spent < escalation.budget {
            return false;
        }

        self.escalation = None;
        self.simulate(task);
        self.pruning = Pruning::EffectiveSimulation;
        true
    }

    /// The number of leaves whose simulation has been computed.
    pub(crate) fn simulated_leaves(&self) -> u64 {
        self.simulations
            .iter()
            .map(|s| u64::from(s.is_some()))
            .sum()
    }

    /// The simulators of leaf `leaf`, if computed.
    pub(crate) fn simulators(&self, leaf: usize) -> Option<&[Bits]> {
        self.simulations[leaf].as_deref()
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
                    self.leaf_bounds(leaf, analysis, prices, size)
                })
            })
            .collect()
    }

    /// The bounds of one leaf with `prices` under `analysis`.
    fn leaf_bounds(
        &self,
        leaf: usize,
        analysis: &Analysis,
        prices: &Prices,
        size: usize,
    ) -> Vec<Bound> {
        let simulators = self
            .simulators(leaf)
            .filter(|_| uses_simulation(self.pruning));
        let base = if let Some(simulators) = simulators {
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
            Pruning::Basic | Pruning::Frontier | Pruning::Simulation | Pruning::Adaptive => base,
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

            if !self.greedy && !self.fold_g && delta > 0 {
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

/// Whether `pruning` compares through leaf simulations.
fn uses_simulation(pruning: Pruning) -> bool {
    matches!(pruning, Pruning::Simulation | Pruning::EffectiveSimulation)
}

/// The leaf parts on leaf `leaf` of all global actions.
fn parts(task: &DecoupledTask, leaf: usize) -> impl Iterator<Item = &LeafPart> + Clone {
    task.globals
        .iter()
        .filter_map(move |g| g.parts[leaf].as_ref())
}

/// The static analysis of leaf `leaf`: anchored states and leaf
/// transitions under any center.
fn analyse(task: &DecoupledTask, leaf: usize) -> Analysis {
    let parts: Vec<&LeafPart> = parts(task, leaf).collect();
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

    Analysis {
        anchored,
        successors,
        predecessors,
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
