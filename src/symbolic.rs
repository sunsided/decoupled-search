//! Decoupled search with symbolic representations: leaf pricing functions
//! as BDD layers (Gnad, Torralba and Hoffmann 2017), small leaves kept
//! enumerated next to them, and optionally a symbolic center that expands
//! sets of center states sharing one pricing function at once.
//! [`crate::Config::symbolic_leaves`] and
//! [`crate::Config::symbolic_center`] state what is supported;
//! `docs/design.md` describes the operations.

mod bdd_leaf;
mod explicit;
mod grouped;

use alloc::borrow::Cow;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Fact, OpId, Operator, Task, Value, VarId};
use symbolic_search::backend::{Backend, BackendResult, Capacity, OutOfMemory, Oxidd, OxiddSet};

use crate::decoupled::{Price, UNREACHED};
use crate::heuristic::fact_offsets;
use crate::search::{Order, with_true_cost};
use crate::{
    Config, DecoupledHeuristic, Error, FactPrices, Factoring, Leaves, MAX_LEAF_STATES, Outcome,
    Plan, Pruning, Statistics,
};
use bdd_leaf::{Encoding, Layers, LeafMove, SymLeaf};
use explicit::{ExplicitLeaf, LocalFacts, Move};

/// A set of states of one factor.
type Set = OxiddSet;

/// Which leaf actions a closure may use, by their center preconditions.
type Active<'a> = &'a dyn Fn(&[(usize, Value)]) -> bool;

/// The smallest and largest node table tried, in nodes.
const NODES: (usize, usize) = (1 << 14, 1 << 24);

/// One compiled leaf.
enum Leaf {
    /// Pricing as BDD layers.
    Symbolic(SymLeaf),
    /// Pricing as one price per enumerated leaf state.
    Explicit(ExplicitLeaf),
}

/// The pricing function of one leaf, in its leaf's representation. Equal
/// pricing functions compare equal (BDDs are canonical).
#[derive(Clone, PartialEq)]
enum Pricing {
    /// BDD layers.
    Symbolic(Layers),
    /// Dense prices, [`UNREACHED`] where not reached.
    Explicit(Vec<Price>),
}

/// A global action's part on one leaf.
enum Part {
    /// Its relation.
    Symbolic(Set),
    /// Its successor table.
    Explicit(Vec<Option<usize>>),
}

/// A global action: an operator that is not a leaf action.
struct Global {
    /// The operator.
    op: OpId,
    /// Its cost.
    cost: Price,
    /// Its center preconditions.
    center_pre: Vec<(usize, Value)>,
    /// Its center effects in operator order (last per slot).
    center_eff: Vec<(usize, Value)>,
    /// Its part on each leaf it touches.
    parts: Vec<Option<Part>>,
}

/// A decoupled state with a concrete center.
#[derive(Clone)]
struct State {
    /// The center values, by center slot.
    center: Vec<Value>,
    /// The pricing function of every leaf.
    leaves: Vec<Pricing>,
}

/// A task compiled for the symbolic search.
struct SymbolicTask {
    /// The BDD manager all symbolic factors share, on disjoint variables.
    bdd: Oxidd,
    /// Initial center values.
    initial_center: Vec<Value>,
    /// The center goal.
    center_goal: Vec<(usize, Value)>,
    /// Domain size of each center slot.
    center_domains: Vec<usize>,
    /// The center's encoding, if the center is symbolic.
    center_encoding: Option<Encoding>,
    /// The leaves, in factoring order.
    leaves: Vec<Leaf>,
    /// The global actions, in operator order.
    globals: Vec<Global>,
    /// Fact offsets for [`FactPrices`].
    fact_offsets: Vec<usize>,
    /// The leaf of every task variable.
    leaf_of: Vec<Option<usize>>,
    /// Center slot or `(leaf, slot)` of every task variable.
    place: Vec<Place>,
    /// Whether leaf moves cost their cost (off for reachability-only
    /// leaves).
    costs: bool,
}

/// Where a task variable lives.
#[derive(Clone, Copy)]
enum Place {
    /// Center slot.
    Center(usize),
    /// Leaf and slot.
    Leaf(usize, usize),
}

/// Runs the search `order` of `config` with symbolic leaves or center.
///
/// # Errors
///
/// As [`Config::search`], plus [`Error::SymbolicUnsupported`] for options
/// and tasks the symbolic search does not cover and [`Error::OutOfNodes`]
/// if the largest node table is too small.
pub(crate) fn run(
    config: &Config,
    task: &Task,
    factoring: &Factoring,
    heuristics: (
        Option<&dyn DecoupledHeuristic>,
        Option<&dyn DecoupledHeuristic>,
    ),
    order: Order,
) -> Result<Outcome, Error> {
    check_supported(config, task, order)?;
    if config.leaves == Leaves::Reachability && order != Order::Greedy {
        return Err(Error::ReachabilityNotOptimal);
    }

    let mut nodes = initial_nodes(task);
    loop {
        let capacity = Capacity {
            nodes,
            cache: nodes / 4,
        };
        let attempt = SymbolicTask::new(task, factoring, capacity, config).and_then(|t| {
            if config.symbolic_center {
                grouped::search(&t, task, config, heuristics.0, order).map_err(Retry::from)
            } else {
                search(&t, task, config, heuristics, order).map_err(Retry::from)
            }
        });
        match attempt {
            Ok(outcome) => return Ok(outcome),
            Err(Retry::Error(e)) => return Err(e),
            Err(Retry::OutOfMemory) if nodes < NODES.1 => nodes *= 4,
            Err(Retry::OutOfMemory) => return Err(Error::OutOfNodes),
        }
    }
}

/// Why compiling or searching failed.
enum Retry {
    /// The node table was too small.
    OutOfMemory,
    /// A task or factoring error.
    Error(Error),
}

impl From<OutOfMemory> for Retry {
    fn from(_: OutOfMemory) -> Self {
        Self::OutOfMemory
    }
}

/// Rejects the options and task features the symbolic search does not
/// cover.
fn check_supported(config: &Config, task: &Task, order: Order) -> Result<(), Error> {
    let unsupported = |option| Err(Error::SymbolicUnsupported { option });
    task.validate().map_err(Error::InvalidTask)?;
    if config.pruning != Pruning::Basic {
        return unsupported("dominance criteria other than Pruning::Basic");
    }
    if config.reduce_leaves {
        return unsupported("Config::reduce_leaves");
    }
    if config.stubborn_sets {
        return unsupported("Config::stubborn_sets");
    }
    if config.symbolic_center && config.fold_g {
        return unsupported("Config::fold_g with a symbolic center");
    }
    if config.symbolic_center && order == Order::Anytime {
        return unsupported("anytime A* with a symbolic center");
    }
    if !task.axioms.is_empty() {
        return unsupported("axioms");
    }
    if task
        .operators
        .iter()
        .any(|op| op.eff.iter().any(|e| !e.conditions.is_empty()))
    {
        return unsupported("conditional effects");
    }

    Ok(())
}

/// A starting node table size for `task`, as `symbolic-search` sizes it.
fn initial_nodes(task: &Task) -> usize {
    let bits: usize = task.variables.iter().map(|v| width(v.domain_size)).sum();
    bits.saturating_mul(1 << 9)
        .saturating_add(
            bits.saturating_mul(task.operators.len())
                .saturating_mul(1 << 6),
        )
        .clamp(NODES.0, NODES.1)
        .next_power_of_two()
}

/// Bits to encode a domain of `size` values.
fn width(size: u32) -> usize {
    usize::try_from(u32::BITS - size.saturating_sub(1).leading_zeros()).expect("fits")
}

/// A value as an index.
fn widen(value: Value) -> usize {
    usize::try_from(value).expect("values fit usize")
}

/// A domain size as an index.
fn domain(task: &Task, var: VarId) -> usize {
    widen(task.variables[var].domain_size)
}

impl SymbolicTask {
    /// Compiles `task` against `factoring` into a manager of `capacity`:
    /// leaves with at least [`Config::symbolic_min_states`] states (all of
    /// them with [`Config::symbolic_leaves`] off: none) symbolic, the rest
    /// enumerated; the center encoded if [`Config::symbolic_center`] is on.
    fn new(
        task: &Task,
        factoring: &Factoring,
        capacity: Capacity,
        config: &Config,
    ) -> Result<Self, Retry> {
        let leaf_of = factoring.leaf_of(task).map_err(Retry::Error)?;
        let center_vars: Vec<VarId> = factoring.center().to_vec();
        let place = places(task, factoring);
        let symbolic = representations(task, factoring, config).map_err(Retry::Error)?;

        let widths = |vars: &[VarId]| {
            vars.iter()
                .map(|&v| width(task.variables[v].domain_size))
                .collect::<Vec<_>>()
        };
        let encoded = factoring
            .leaves()
            .iter()
            .zip(&symbolic)
            .filter_map(|(vars, s)| s.then_some(vars.as_slice()))
            .chain(config.symbolic_center.then_some(center_vars.as_slice()));
        let bits: usize = encoded.map(|vars| widths(vars).iter().sum::<usize>()).sum();
        let bdd = Oxidd::new(u32::try_from(2 * bits).expect("bits fit u32"), capacity)?;

        let state = task.initial_state();
        let mut next_var = 0u32;
        let mut leaves = Vec::new();
        for (leaf, vars) in factoring.leaves().iter().enumerate() {
            let domains: Vec<usize> = vars.iter().map(|&v| domain(task, v)).collect();
            let initial: Vec<usize> = vars.iter().map(|&v| widen(state.0[v])).collect();
            let goal = leaf_facts(&place, leaf, &task.goal);
            leaves.push(if symbolic[leaf] {
                let encoding = Encoding::new(&bdd, &widths(vars), &mut next_var)?;
                Leaf::Symbolic(SymLeaf::new(&bdd, encoding, &domains, &initial, &goal)?)
            } else {
                Leaf::Explicit(ExplicitLeaf::new(domains, &initial, &goal))
            });
        }
        let center_encoding = config
            .symbolic_center
            .then(|| Encoding::new(&bdd, &widths(&center_vars), &mut next_var))
            .transpose()?;

        let mut globals = Vec::new();
        for (op, operator) in task.operators.iter().enumerate() {
            if let Some(global) = classify(&bdd, &place, &mut leaves, op, operator)? {
                globals.push(global);
            }
        }
        for leaf in &mut leaves {
            if let Leaf::Symbolic(l) = leaf {
                l.group_moves(&bdd)?;
            }
        }

        Ok(Self {
            bdd,
            initial_center: center_vars.iter().map(|&v| state.0[v]).collect(),
            center_goal: center_facts(&place, &task.goal),
            center_domains: center_vars.iter().map(|&v| domain(task, v)).collect(),
            center_encoding,
            leaves,
            globals,
            fact_offsets: fact_offsets(task),
            leaf_of,
            place,
            costs: config.leaves != Leaves::Reachability,
        })
    }

    /// Leaf `leaf` closed from `seeds` over the leaf actions `active`
    /// accepts.
    fn close(&self, leaf: usize, seeds: &Pricing, active: Active<'_>) -> BackendResult<Pricing> {
        Ok(match (&self.leaves[leaf], seeds) {
            (Leaf::Symbolic(l), Pricing::Symbolic(s)) => {
                Pricing::Symbolic(l.close(&self.bdd, s, active, self.costs)?)
            }
            (Leaf::Explicit(l), Pricing::Explicit(s)) => {
                Pricing::Explicit(l.close(s, active, self.costs).0)
            }
            _ => unreachable!("a pricing matches its leaf"),
        })
    }

    /// The seed pricing of leaf `leaf`: its initial leaf state at 0.
    fn initial_seeds(&self, leaf: usize) -> Pricing {
        match &self.leaves[leaf] {
            Leaf::Symbolic(l) => Pricing::Symbolic(vec![(0, l.initial.clone())]),
            Leaf::Explicit(l) => {
                let mut prices = vec![UNREACHED; l.size];
                prices[l.initial] = 0;
                Pricing::Explicit(prices)
            }
        }
    }

    /// `pricing` of `leaf` carried across `part`; `None` if no leaf state
    /// survives.
    fn transfer(
        &self,
        leaf: usize,
        part: &Part,
        pricing: &Pricing,
    ) -> BackendResult<Option<Pricing>> {
        Ok(match (&self.leaves[leaf], part, pricing) {
            (Leaf::Symbolic(l), Part::Symbolic(r), Pricing::Symbolic(p)) => {
                l.transfer(&self.bdd, r, p)?.map(Pricing::Symbolic)
            }
            (Leaf::Explicit(l), Part::Explicit(t), Pricing::Explicit(p)) => {
                l.transfer(t, p).map(Pricing::Explicit)
            }
            _ => unreachable!("a part and pricing match their leaf"),
        })
    }

    /// The pricings of `leaves` carried across global action `index`;
    /// `None` if some leaf it touches keeps no leaf state.
    fn carried(&self, index: usize, leaves: &[Pricing]) -> BackendResult<Option<Vec<Pricing>>> {
        let mut seeds = Vec::with_capacity(leaves.len());
        for (leaf, (part, pricing)) in self.globals[index].parts.iter().zip(leaves).enumerate() {
            match part {
                None => seeds.push(pricing.clone()),
                Some(part) => match self.transfer(leaf, part, pricing)? {
                    Some(moved) => seeds.push(moved),
                    None => return Ok(None),
                },
            }
        }

        Ok(Some(seeds))
    }

    /// Every leaf of `seeds` closed under `active`.
    fn closed(&self, seeds: &[Pricing], active: Active<'_>) -> BackendResult<Vec<Pricing>> {
        seeds
            .iter()
            .enumerate()
            .map(|(l, s)| self.close(l, s, active))
            .collect()
    }

    /// The cheapest goal price of leaf `leaf` in `pricing`.
    fn leaf_goal_price(&self, leaf: usize, pricing: &Pricing) -> BackendResult<Option<Price>> {
        Ok(match (&self.leaves[leaf], pricing) {
            (Leaf::Symbolic(l), Pricing::Symbolic(p)) => SymLeaf::min_price(&self.bdd, p, &l.goal)?,
            (Leaf::Explicit(l), Pricing::Explicit(p)) => ExplicitLeaf::min_price(p, |x| l.goal[x]),
            _ => unreachable!("a pricing matches its leaf"),
        })
    }

    /// The sum of the leaves' cheapest goal prices; `None` if a leaf
    /// reaches no goal state.
    fn leaves_goal_price(&self, leaves: &[Pricing]) -> BackendResult<Option<Price>> {
        let mut sum: Price = 0;
        for (leaf, pricing) in leaves.iter().enumerate() {
            match self.leaf_goal_price(leaf, pricing)? {
                Some(p) => sum = sum.saturating_add(p),
                None => return Ok(None),
            }
        }

        Ok(Some(sum))
    }

    /// Fact prices: 0 for the center facts `center_has` accepts, the
    /// cheapest price over reached leaf states for leaf facts.
    fn fact_prices(
        &self,
        center_has: &dyn Fn(usize, usize) -> BackendResult<bool>,
        leaves: &[Pricing],
    ) -> BackendResult<FactPrices<'_>> {
        let mut prices = vec![UNREACHED; *self.fact_offsets.last().expect("final entry")];
        for (var, place) in self.place.iter().enumerate() {
            let base = self.fact_offsets[var];
            let size = self.fact_offsets[var + 1] - base;
            for value in 0..size {
                let price = match *place {
                    Place::Center(slot) => center_has(slot, value)?.then_some(0),
                    Place::Leaf(leaf, slot) => match (&self.leaves[leaf], &leaves[leaf]) {
                        (Leaf::Symbolic(l), Pricing::Symbolic(p)) => {
                            SymLeaf::min_price(&self.bdd, p, &l.facts[slot][value])?
                        }
                        (Leaf::Explicit(l), Pricing::Explicit(p)) => {
                            ExplicitLeaf::min_price(p, |x| l.satisfies(x, &[(slot, value)]))
                        }
                        _ => unreachable!("a pricing matches its leaf"),
                    },
                };
                if let Some(p) = price {
                    prices[base + value] = p;
                }
            }
        }

        Ok(FactPrices {
            offsets: Cow::Borrowed(&self.fact_offsets),
            leaves: Cow::Borrowed(&self.leaf_of),
            prices,
        })
    }

    /// Whether the leaves `t` (reached at `g_t`) dominate the leaves `n`
    /// (at `g_n`) under [`Pruning::Basic`], as the explicit criterion
    /// decides it.
    fn dominates(
        &self,
        (t, g_t): (&[Pricing], Price),
        (n, g_n): (&[Pricing], Price),
        greedy: bool,
        fold_g: bool,
    ) -> BackendResult<bool> {
        let mut budget: i128 = 0;
        for (new, stored) in n.iter().zip(t) {
            let delta = match (stored, new) {
                (Pricing::Symbolic(a), Pricing::Symbolic(b)) => SymLeaf::slack(&self.bdd, a, b)?,
                (Pricing::Explicit(a), Pricing::Explicit(b)) => ExplicitLeaf::slack(a, b),
                _ => unreachable!("pricings of one leaf match"),
            };
            let Some(delta) = delta else {
                return Ok(false);
            };
            if !greedy && !fold_g && delta > 0 {
                return Ok(false);
            }

            budget = budget.saturating_add(delta);
        }

        if greedy {
            return Ok(true);
        }

        let (g_t, g_n) = (i128::from(g_t), i128::from(g_n));
        Ok(if fold_g {
            g_t.saturating_add(budget) <= g_n
        } else {
            g_t <= g_n
        })
    }

    /// Per leaf, the cheapest leaf-action cost from the initial leaf state
    /// to a goal leaf state with global parts free and center preconditions
    /// ignored, summed; [`Price::MAX`] if some leaf never reaches its goal.
    fn min_goal_price(&self) -> BackendResult<Price> {
        let mut sum: Price = 0;
        for (l, leaf) in self.leaves.iter().enumerate() {
            let parts = self.globals.iter().filter_map(|g| g.parts[l].as_ref());
            let best = match leaf {
                Leaf::Symbolic(leaf) => {
                    let free: Vec<&Set> = parts
                        .map(|p| match p {
                            Part::Symbolic(r) => r,
                            Part::Explicit(_) => unreachable!("parts match their leaf"),
                        })
                        .collect();
                    leaf.cheapest_goal_cost(&self.bdd, &free)?
                }
                Leaf::Explicit(leaf) => {
                    let free: Vec<&[Option<usize>]> = parts
                        .map(|p| match p {
                            Part::Explicit(t) => t.as_slice(),
                            Part::Symbolic(_) => unreachable!("parts match their leaf"),
                        })
                        .collect();
                    leaf.cheapest_goal_cost(&free)
                }
            };
            sum = sum.saturating_add(best.unwrap_or(Price::MAX));
        }

        Ok(sum)
    }

    /// The sequential plan along a solution path with concrete centers
    /// `centers`, global actions `actions` into each stage (`None` for the
    /// first) and leaf pricings `pricings`: per stage the global action,
    /// then every leaf's leaf actions.
    fn plan(
        &self,
        centers: &[Vec<Value>],
        actions: &[Option<usize>],
        pricings: &[&[Pricing]],
        cost: Price,
    ) -> BackendResult<Plan> {
        let leaf_ops = (0..self.leaves.len())
            .map(|leaf| self.leaf_stages(leaf, centers, actions, pricings))
            .collect::<BackendResult<Vec<_>>>()?;

        let mut ops = Vec::new();
        for (stage, action) in actions.iter().enumerate() {
            if let Some(action) = action {
                ops.push(self.globals[*action].op);
            }
            for stages in &leaf_ops {
                ops.extend_from_slice(&stages[stage]);
            }
        }

        Ok(Plan { ops, cost })
    }

    /// The leaf actions of leaf `leaf` in every stage of a solution path.
    fn leaf_stages(
        &self,
        leaf: usize,
        centers: &[Vec<Value>],
        actions: &[Option<usize>],
        pricings: &[&[Pricing]],
    ) -> BackendResult<Vec<Vec<OpId>>> {
        let part = |stage: usize| actions[stage].and_then(|a| self.globals[a].parts[leaf].as_ref());
        match &self.leaves[leaf] {
            Leaf::Symbolic(l) => {
                let parts: Vec<Option<&Set>> = (0..actions.len())
                    .map(|k| {
                        part(k).map(|p| match p {
                            Part::Symbolic(r) => r,
                            Part::Explicit(_) => unreachable!("parts match their leaf"),
                        })
                    })
                    .collect();
                let layers: Vec<&Layers> = pricings
                    .iter()
                    .map(|p| match &p[leaf] {
                        Pricing::Symbolic(l) => l,
                        Pricing::Explicit(_) => unreachable!("pricings match their leaf"),
                    })
                    .collect();
                l.stages(&self.bdd, self.costs, centers, &parts, &layers)
            }
            Leaf::Explicit(l) => Ok(self.explicit_stages(l, leaf, centers, &part, pricings)),
        }
    }

    /// [`SymbolicTask::leaf_stages`] for an enumerated leaf: closure
    /// parents per stage, then the source of the carried seed.
    fn explicit_stages<'a>(
        &'a self,
        l: &ExplicitLeaf,
        leaf: usize,
        centers: &[Vec<Value>],
        part: &dyn Fn(usize) -> Option<&'a Part>,
        pricings: &[&[Pricing]],
    ) -> Vec<Vec<OpId>> {
        let dense = |stage: usize| match &pricings[stage][leaf] {
            Pricing::Explicit(p) => p.as_slice(),
            Pricing::Symbolic(_) => unreachable!("pricings match their leaf"),
        };
        let table = |stage: usize| {
            part(stage).map(|p| match p {
                Part::Explicit(t) => t.as_slice(),
                Part::Symbolic(_) => unreachable!("parts match their leaf"),
            })
        };
        let mut target = l
            .cheapest_goal(dense(centers.len() - 1))
            .expect("a solution reaches a leaf goal state");
        let mut stages = vec![Vec::new(); centers.len()];
        for stage in (0..centers.len()).rev() {
            let seeds = if stage == 0 {
                let mut s = vec![UNREACHED; l.size];
                s[l.initial] = 0;
                s
            } else {
                match table(stage) {
                    None => dense(stage - 1).to_vec(),
                    Some(t) => l
                        .transfer(t, dense(stage - 1))
                        .expect("the path applied this action"),
                }
            };
            let center = &centers[stage];
            let active = |pre: &[(usize, Value)]| pre.iter().all(|&(s, v)| center[s] == v);
            let (_, parents) = l.close(&seeds, &active, self.costs);
            let mut rank = target;
            while let Some((from, index)) = parents[rank] {
                stages[stage].push(l.moves[index].op);
                rank = from;
            }
            stages[stage].reverse();
            target = rank;
            if let (true, Some(t)) = (stage > 0, table(stage)) {
                target = l.source(t, dense(stage - 1), rank, seeds[rank]);
            }
        }

        stages
    }
}

/// The center slot or `(leaf, slot)` of every task variable.
fn places(task: &Task, factoring: &Factoring) -> Vec<Place> {
    let mut place = vec![Place::Center(0); task.num_variables()];
    for (slot, &var) in factoring.center().iter().enumerate() {
        place[var] = Place::Center(slot);
    }
    for (leaf, vars) in factoring.leaves().iter().enumerate() {
        for (slot, &var) in vars.iter().enumerate() {
            place[var] = Place::Leaf(leaf, slot);
        }
    }

    place
}

/// Per leaf, whether it is symbolic: [`Config::symbolic_leaves`] on and at
/// least [`Config::symbolic_min_states`] states.
///
/// # Errors
///
/// [`Error::LeafTooLarge`] for an enumerated leaf above
/// [`MAX_LEAF_STATES`].
fn representations(
    task: &Task,
    factoring: &Factoring,
    config: &Config,
) -> Result<Vec<bool>, Error> {
    let mut symbolic = Vec::new();
    for (leaf, vars) in factoring.leaves().iter().enumerate() {
        let size = vars
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(domain(task, v)));
        let is_symbolic =
            config.symbolic_leaves && size.is_none_or(|n| n >= config.symbolic_min_states);
        if !is_symbolic && size.is_none_or(|n| n > MAX_LEAF_STATES) {
            return Err(Error::LeafTooLarge { leaf });
        }
        symbolic.push(is_symbolic);
    }

    Ok(symbolic)
}

/// Adds leaf action `op` with its leaf part `facts` to `leaf`.
fn add_move(
    bdd: &Oxidd,
    leaf: &mut Leaf,
    op: OpId,
    cost: Price,
    center_pre: Vec<(usize, Value)>,
    facts: &LocalFacts,
) -> BackendResult<()> {
    match leaf {
        Leaf::Symbolic(l) => {
            let relation = l.encoding.relation(bdd, &facts.pre, &facts.eff)?;
            l.moves.push(LeafMove {
                op,
                cost,
                center_pre,
                relation,
            });
        }
        Leaf::Explicit(l) => {
            let next = l.table(facts);
            l.moves.push(Move {
                op,
                cost,
                center_pre,
                next,
            });
        }
    }

    Ok(())
}

/// Adds operator `op` to its leaf as a move if it is a leaf action (it sets
/// no center variable and touches exactly one leaf, which it sets), as the
/// explicit compilation decides; otherwise returns it as a global action.
fn classify(
    bdd: &Oxidd,
    place: &[Place],
    leaves: &mut [Leaf],
    op: OpId,
    operator: &Operator,
) -> BackendResult<Option<Global>> {
    let cost = Price::from(operator.cost);
    let mut eff: Vec<Fact> = Vec::new();
    for e in &operator.eff {
        eff.retain(|f| f.var != e.fact.var);
        eff.push(e.fact);
    }

    let center_pre = center_facts(place, &operator.pre);
    let center_eff = center_facts(place, &eff);
    let local = |l: usize| LocalFacts {
        pre: leaf_facts(place, l, &operator.pre),
        eff: leaf_facts(place, l, &eff),
    };
    let touched: Vec<usize> = (0..leaves.len())
        .filter(|&l| {
            operator
                .pre
                .iter()
                .chain(&eff)
                .any(|f| leaf_is(place, f, l))
        })
        .collect();
    if let [leaf] = touched[..]
        && center_eff.is_empty()
        && eff.iter().any(|f| leaf_is(place, f, leaf))
    {
        add_move(bdd, &mut leaves[leaf], op, cost, center_pre, &local(leaf))?;
        return Ok(None);
    }

    let mut parts = Vec::with_capacity(leaves.len());
    for (l, leaf) in leaves.iter().enumerate() {
        if !touched.contains(&l) {
            parts.push(None);
            continue;
        }
        let facts = local(l);
        parts.push(Some(match leaf {
            Leaf::Symbolic(s) => Part::Symbolic(s.encoding.relation(bdd, &facts.pre, &facts.eff)?),
            Leaf::Explicit(e) => Part::Explicit(e.table(&facts)),
        }));
    }

    Ok(Some(Global {
        op,
        cost,
        center_pre,
        center_eff,
        parts,
    }))
}

/// The facts of `facts` on center variables, as `(slot, value)`.
fn center_facts(place: &[Place], facts: &[Fact]) -> Vec<(usize, Value)> {
    facts
        .iter()
        .filter_map(|f| match place[f.var] {
            Place::Center(slot) => Some((slot, f.value)),
            Place::Leaf(..) => None,
        })
        .collect()
}

/// The facts of `facts` on leaf `leaf`, as `(slot, value)`.
fn leaf_facts(place: &[Place], leaf: usize, facts: &[Fact]) -> Vec<(usize, usize)> {
    facts
        .iter()
        .filter_map(|f| match place[f.var] {
            Place::Leaf(l, slot) if l == leaf => Some((slot, widen(f.value))),
            _ => None,
        })
        .collect()
}

/// Whether fact `f` lies on leaf `leaf`.
fn leaf_is(place: &[Place], f: &Fact, leaf: usize) -> bool {
    matches!(place[f.var], Place::Leaf(l, _) if l == leaf)
}

/// Whether `center` satisfies `facts`.
fn holds(facts: &[(usize, Value)], center: &[Value]) -> bool {
    facts.iter().all(|&(s, v)| center[s] == v)
}

/// A search node with a concrete center.
struct Node {
    /// The decoupled state.
    state: State,
    /// Cost of the global actions so far.
    g: Price,
    /// Parent node and global action index.
    parent: Option<(usize, usize)>,
}

/// Best-first search over decoupled states with concrete centers; the same
/// loop as the explicit search under [`Pruning::Basic`], so both expand
/// the same states in the same order.
fn search(
    task: &SymbolicTask,
    original: &Task,
    config: &Config,
    (heuristic, root): (
        Option<&dyn DecoupledHeuristic>,
        Option<&dyn DecoupledHeuristic>,
    ),
    order: Order,
) -> BackendResult<Outcome> {
    let min_goal_price = if order == Order::Anytime {
        task.min_goal_price()?
    } else {
        0
    };
    let mut search = Search {
        task,
        heuristic,
        root,
        order,
        fold_g: config.fold_g,
        min_goal_price,
        nodes: Vec::new(),
        open: BinaryHeap::new(),
        by_center: BTreeMap::new(),
        best: None,
        statistics: Statistics::default(),
    };
    search.run()?;

    let plan = match search.best {
        None => None,
        Some((cost, id)) => {
            let path = search.path(id);
            let centers: Vec<Vec<Value>> = path.iter().map(|(_, s)| s.center.clone()).collect();
            let actions: Vec<Option<usize>> = path.iter().map(|(a, _)| *a).collect();
            let pricings: Vec<&[Pricing]> = path.iter().map(|(_, s)| s.leaves.as_slice()).collect();
            let plan = task.plan(&centers, &actions, &pricings, cost)?;
            Some(with_true_cost(original, plan, !task.costs))
        }
    };

    Ok(Outcome {
        plan,
        statistics: search.statistics,
    })
}

/// Search bookkeeping, as in the explicit search.
struct Search<'t> {
    /// The task.
    task: &'t SymbolicTask,
    /// The heuristic; `None` for `h = 0`.
    heuristic: Option<&'t dyn DecoupledHeuristic>,
    /// The root heuristic of anytime A*.
    root: Option<&'t dyn DecoupledHeuristic>,
    /// The open-list order.
    order: Order,
    /// [`Config::fold_g`].
    fold_g: bool,
    /// The anytime A* bound on every goal price; 0 otherwise.
    min_goal_price: Price,
    /// Every stored node.
    nodes: Vec<Node>,
    /// Node ids by key.
    open: BinaryHeap<Reverse<(Price, usize)>>,
    /// Node ids per center state.
    by_center: BTreeMap<Vec<Value>, Vec<usize>>,
    /// Cheapest solution so far and its node.
    best: Option<(Price, usize)>,
    /// Counters.
    statistics: Statistics,
}

impl Search<'_> {
    /// The initial decoupled state.
    fn initial(&self) -> BackendResult<State> {
        let center = self.task.initial_center.clone();
        let seeds: Vec<Pricing> = (0..self.task.leaves.len())
            .map(|l| self.task.initial_seeds(l))
            .collect();
        let active = |pre: &[(usize, Value)]| holds(pre, &center);
        let leaves = self.task.closed(&seeds, &active)?;
        Ok(State { center, leaves })
    }

    /// The successor of `state` under global action `index`.
    fn successor(&self, index: usize, state: &State) -> BackendResult<Option<State>> {
        let g = &self.task.globals[index];
        if !holds(&g.center_pre, &state.center) {
            return Ok(None);
        }
        let Some(seeds) = self.task.carried(index, &state.leaves)? else {
            return Ok(None);
        };

        let mut center = state.center.clone();
        for &(slot, value) in &g.center_eff {
            center[slot] = value;
        }
        let active = |pre: &[(usize, Value)]| holds(pre, &center);
        let leaves = self.task.closed(&seeds, &active)?;
        Ok(Some(State { center, leaves }))
    }

    /// The fact prices of `state`.
    fn fact_prices(&self, state: &State) -> BackendResult<FactPrices<'_>> {
        let center = &state.center;
        self.task.fact_prices(
            &|slot, value| Ok(widen(center[slot]) == value),
            &state.leaves,
        )
    }

    /// Inserts the initial state and expands nodes until done.
    fn run(&mut self) -> BackendResult<()> {
        let initial = self.initial()?;
        self.insert(initial, 0, None)?;
        while let Some(Reverse((key, id))) = self.open.pop() {
            if self.is_done(key) {
                break;
            }

            self.statistics.expanded += 1;
            let g = self.nodes[id].g;
            let mut successors = Vec::new();
            for i in 0..self.task.globals.len() {
                if let Some(s) = self.successor(i, &self.nodes[id].state)? {
                    successors.push((i, s));
                }
            }
            for (action, state) in successors {
                let g = g.saturating_add(self.task.globals[action].cost);
                self.insert(state, g, Some((id, action)))?;
            }
        }

        Ok(())
    }

    /// Whether the search stops before a node with key `key`.
    fn is_done(&self, key: Price) -> bool {
        match self.order {
            Order::Cost => self.best.is_some_and(|(c, _)| key >= c),
            Order::Anytime => self
                .best
                .is_some_and(|(c, _)| key.saturating_add(self.min_goal_price) >= c),
            Order::Greedy => self.best.is_some(),
        }
    }

    /// The open-list key of `state` at `g` with heuristic value `h`; `None`
    /// if the root heuristic declares it a dead end.
    fn key(&self, state: &State, g: Price, h: Price) -> BackendResult<Option<Price>> {
        Ok(match self.order {
            Order::Cost => Some(g.saturating_add(h)),
            Order::Greedy => Some(h),
            Order::Anytime => {
                let root = match self.root {
                    None => Some(0),
                    Some(r) => r.estimate(&self.fact_prices(state)?.reached_for_free()),
                };
                root.map(|r| g.saturating_add(r))
            }
        })
    }

    /// Whether a stored node with the same center dominates `state` at `g`.
    fn is_dominated(&self, state: &State, g: Price) -> BackendResult<bool> {
        let greedy = self.order == Order::Greedy;
        for &id in self.by_center.get(&state.center).into_iter().flatten() {
            let node = &self.nodes[id];
            let stored = (node.state.leaves.as_slice(), node.g);
            if self
                .task
                .dominates(stored, (&state.leaves, g), greedy, self.fold_g)?
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Adds `state` at `g` unless it is dominated, a dead end or above the
    /// anytime bound, and records it if it is the best solution so far.
    fn insert(
        &mut self,
        state: State,
        g: Price,
        parent: Option<(usize, usize)>,
    ) -> BackendResult<()> {
        self.statistics.generated += 1;
        if self.is_dominated(&state, g)? {
            self.statistics.dominated += 1;
            return Ok(());
        }

        let estimate = match self.heuristic {
            None => Some(0),
            Some(h) => h.estimate(&self.fact_prices(&state)?),
        };
        let Some(h) = estimate else {
            self.statistics.dead_ends += 1;
            return Ok(());
        };
        let Some(key) = self.key(&state, g, h)? else {
            self.statistics.dead_ends += 1;
            return Ok(());
        };

        if self.order == Order::Anytime && self.best.is_some_and(|(b, _)| g.saturating_add(h) >= b)
        {
            self.statistics.bound_pruned += 1;
            return Ok(());
        }

        let id = self.nodes.len();
        if holds(&self.task.center_goal, &state.center)
            && let Some(rest) = self.task.leaves_goal_price(&state.leaves)?
        {
            let cost = g.saturating_add(rest);
            if self.best.is_none_or(|(b, _)| cost < b) {
                self.best = Some((cost, id));
            }
        }

        self.by_center
            .entry(state.center.clone())
            .or_default()
            .push(id);
        self.open.push(Reverse((key, id)));
        self.nodes.push(Node { state, g, parent });
        self.statistics.stored += 1;
        Ok(())
    }

    /// The path to node `id`, initial node first.
    fn path(&self, mut id: usize) -> Vec<(Option<usize>, &State)> {
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
