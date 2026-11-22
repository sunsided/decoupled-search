//! Decoupled search with symbolic leaves (Gnad, Torralba and Hoffmann
//! 2017): every leaf's pricing function is a list of BDD layers, one
//! per price, instead of a price per enumerated leaf state. The decoupled
//! state space is the explicit one; [`crate::Config::symbolic_leaves`]
//! states what is supported, and `docs/design.md` describes the operations.

use alloc::borrow::Cow;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Fact, OpId, Task, Value, VarId};
use symbolic_search::backend::{
    Backend, BackendResult, Capacity, OutOfMemory, Oxidd, OxiddRenaming, OxiddSet, OxiddVarSet,
};

use crate::decoupled::{Price, UNREACHED};
use crate::heuristic::fact_offsets;
use crate::search::{Order, with_true_cost};
use crate::{Config, DecoupledHeuristic, Error, FactPrices, Factoring, Leaves, Plan, Pruning};
use crate::{Outcome, Statistics};

/// A set of leaf states.
type Set = OxiddSet;

/// The pricing function of one leaf: `(price, leaf states)`, ascending in
/// price, pairwise disjoint, never empty, each state at its cheapest price.
type Layers = Vec<(Price, Set)>;

/// A leaf closure with the 0-cost rounds kept apart: `(price, depth,
/// leaf states)` in the order found, for plan reconstruction.
type Rounds = Vec<(Price, usize, Set)>;

/// The smallest and largest node table tried, in nodes.
const NODES: (usize, usize) = (1 << 14, 1 << 24);

/// The binary encoding of one leaf: interleaved current and next bits.
struct Encoding {
    /// Current-state BDD variables of each leaf variable, least significant
    /// bit first.
    bits: Vec<Vec<u32>>,
    /// All current-state variables of the leaf.
    cur: OxiddVarSet,
    /// All next-state variables of the leaf.
    next: OxiddVarSet,
    /// Next to current.
    to_cur: OxiddRenaming,
    /// Current to next.
    to_next: OxiddRenaming,
    /// Every current-state variable, for singleton cubes.
    cur_vars: Vec<u32>,
}

/// A leaf action on its leaf.
struct LeafMove {
    /// The operator.
    op: OpId,
    /// Its cost.
    cost: Price,
    /// Its center preconditions, as `(center slot, value)`.
    center_pre: Vec<(usize, Value)>,
    /// Its transition relation on the leaf.
    relation: Set,
}

/// The leaf actions of one leaf with equal cost and center preconditions,
/// merged into one relation.
struct Group {
    /// The shared cost.
    cost: Price,
    /// The shared center preconditions.
    center_pre: Vec<(usize, Value)>,
    /// The union of the members' relations.
    relation: Set,
}

/// One compiled leaf.
struct Leaf {
    /// The encoding.
    encoding: Encoding,
    /// Its leaf actions.
    moves: Vec<LeafMove>,
    /// Its leaf actions grouped for closure.
    groups: Vec<Group>,
    /// The initial leaf state.
    initial: Set,
    /// The leaf states satisfying the leaf goal (all if it has none).
    goal: Set,
    /// The leaf states satisfying each fact, by slot and value.
    facts: Vec<Vec<Set>>,
}

/// A global action: an operator that is not a leaf action.
struct Global {
    /// The operator.
    op: OpId,
    /// Its cost.
    cost: Price,
    /// Its center preconditions.
    center_pre: Vec<(usize, Value)>,
    /// Its center effects in operator order.
    center_eff: Vec<(usize, Value)>,
    /// Its relation on each leaf it touches.
    parts: Vec<Option<Set>>,
}

/// A decoupled state with symbolic leaves.
#[derive(Clone)]
struct State {
    /// The center values, by center slot.
    center: Vec<Value>,
    /// The pricing function of every leaf.
    leaves: Vec<Layers>,
}

/// A task compiled for symbolic leaves.
struct SymbolicTask {
    /// The BDD manager all leaves share (each on its own variables).
    bdd: Oxidd,
    /// Initial center values.
    initial_center: Vec<Value>,
    /// The center goal.
    center_goal: Vec<(usize, Value)>,
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
    /// Whether leaf costs are ignored (every reached state at price 0).
    reachability_only: bool,
}

/// Where a task variable lives.
#[derive(Clone, Copy)]
enum Place {
    /// Center slot.
    Center(usize),
    /// Leaf and slot.
    Leaf(usize, usize),
}

/// Runs the search `order` of `config` with symbolic leaves.
///
/// # Errors
///
/// As [`Config::search`], plus [`Error::SymbolicUnsupported`] for options
/// and tasks symbolic leaves do not cover and [`Error::OutOfNodes`] if the
/// largest node table is too small.
pub(crate) fn run(
    config: &Config,
    task: &Task,
    factoring: &Factoring,
    (heuristic, root): (
        Option<&dyn DecoupledHeuristic>,
        Option<&dyn DecoupledHeuristic>,
    ),
    order: Order,
) -> Result<Outcome, Error> {
    check_supported(config, task)?;
    if config.leaves == Leaves::Reachability && order != Order::Greedy {
        return Err(Error::ReachabilityNotOptimal);
    }

    let mut nodes = initial_nodes(task);
    loop {
        let capacity = Capacity {
            nodes,
            cache: nodes / 4,
        };
        let attempt = SymbolicTask::new(task, factoring, capacity, config.leaves)
            .and_then(|t| search(&t, task, config, (heuristic, root), order).map_err(Retry::from));
        match attempt {
            Ok(outcome) => return Ok(outcome),
            Err(Retry::Error(e)) => return Err(e),
            Err(Retry::OutOfMemory) if nodes < NODES.1 => nodes *= 4,
            Err(Retry::OutOfMemory) => return Err(Error::OutOfNodes),
        }
    }
}

/// Why compiling failed.
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

/// Rejects the options and task features symbolic leaves do not cover.
fn check_supported(config: &Config, task: &Task) -> Result<(), Error> {
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

impl SymbolicTask {
    /// Compiles `task` against `factoring` into a manager of `capacity`.
    fn new(
        task: &Task,
        factoring: &Factoring,
        capacity: Capacity,
        leaves_mode: Leaves,
    ) -> Result<Self, Retry> {
        let leaf_of = factoring.leaf_of(task).map_err(Retry::Error)?;
        let center_vars: Vec<VarId> = factoring.center().to_vec();
        let mut place = vec![Place::Center(0); task.num_variables()];
        for (slot, &var) in center_vars.iter().enumerate() {
            place[var] = Place::Center(slot);
        }
        for (leaf, vars) in factoring.leaves().iter().enumerate() {
            for (slot, &var) in vars.iter().enumerate() {
                place[var] = Place::Leaf(leaf, slot);
            }
        }

        let total_bits: usize = factoring
            .leaves()
            .iter()
            .flatten()
            .map(|&v| width(task.variables[v].domain_size))
            .sum();
        let bdd = Oxidd::new(
            u32::try_from(2 * total_bits).expect("bits fit u32"),
            capacity,
        )?;

        let mut next_var = 0u32;
        let mut compiled = Vec::new();
        for (leaf, vars) in factoring.leaves().iter().enumerate() {
            let encoding = Encoding::new(&bdd, task, vars, &mut next_var)?;
            compiled.push(Leaf::new(&bdd, task, &place, leaf, vars, encoding)?);
        }

        let mut globals = Vec::new();
        for (op, operator) in task.operators.iter().enumerate() {
            if let Some(global) = classify(&bdd, &place, &mut compiled, op, operator)? {
                globals.push(global);
            }
        }

        for leaf in &mut compiled {
            leaf.group_moves(&bdd)?;
        }

        let initial = task.initial_state();
        Ok(Self {
            bdd,
            initial_center: center_vars.iter().map(|&v| initial.0[v]).collect(),
            center_goal: center_facts(&place, &task.goal),
            leaves: compiled,
            globals,
            fact_offsets: fact_offsets(task),
            leaf_of,
            place,
            reachability_only: leaves_mode == Leaves::Reachability,
        })
    }

    /// The cost of a leaf move in the closure.
    fn effective(&self, cost: Price) -> Price {
        if self.reachability_only { 0 } else { cost }
    }

    /// The image of `set` under `relation` on `leaf`.
    fn image(&self, leaf: usize, set: &Set, relation: &Set) -> BackendResult<Set> {
        let e = &self.leaves[leaf].encoding;
        let next = self.bdd.and_exists(set, relation, &e.cur)?;
        self.bdd.rename(&next, &e.to_cur)
    }

    /// The preimage of `set` under `relation` on `leaf`.
    fn preimage(&self, leaf: usize, set: &Set, relation: &Set) -> BackendResult<Set> {
        let e = &self.leaves[leaf].encoding;
        let next = self.bdd.rename(set, &e.to_next)?;
        self.bdd.and_exists(&next, relation, &e.next)
    }

    /// `a \ b`.
    fn minus(&self, a: &Set, b: &Set) -> BackendResult<Set> {
        let not_b = self.bdd.not(b)?;
        self.bdd.and(a, &not_b)
    }

    /// Whether `a` and `b` share a leaf state.
    fn meets(&self, a: &Set, b: &Set) -> BackendResult<bool> {
        Ok(!self.bdd.is_empty(&self.bdd.and(a, b)?))
    }

    /// Leaf reachability of `leaf` under `center` from `seeds`: symbolic
    /// Dijkstra over the leaf actions whose center preconditions hold, with
    /// the 0-cost rounds of each price kept apart.
    fn close(&self, leaf: usize, center: &[Value], seeds: &Layers) -> BackendResult<Rounds> {
        let active: Vec<&Group> = self.leaves[leaf]
            .groups
            .iter()
            .filter(|g| g.center_pre.iter().all(|&(s, v)| center[s] == v))
            .collect();
        let mut open: BTreeMap<Price, Set> = seeds.iter().cloned().collect();
        let mut closed = self.bdd.constant(false);
        let mut rounds = Vec::new();
        while let Some((price, set)) = open.pop_first() {
            let mut frontier = self.minus(&set, &closed)?;
            let mut layer = self.bdd.constant(false);
            let mut depth = 0;
            while !self.bdd.is_empty(&frontier) {
                closed = self.bdd.or(&closed, &frontier)?;
                layer = self.bdd.or(&layer, &frontier)?;
                rounds.push((price, depth, frontier.clone()));
                let mut found = self.bdd.constant(false);
                for g in active.iter().filter(|g| self.effective(g.cost) == 0) {
                    found = self
                        .bdd
                        .or(&found, &self.image(leaf, &frontier, &g.relation)?)?;
                }
                frontier = self.minus(&found, &closed)?;
                depth += 1;
            }

            if self.bdd.is_empty(&layer) {
                continue;
            }

            for g in &active {
                let cost = self.effective(g.cost);
                if cost == 0 {
                    continue;
                }

                let found = self.minus(&self.image(leaf, &layer, &g.relation)?, &closed)?;
                if self.bdd.is_empty(&found) {
                    continue;
                }

                let at = price.saturating_add(cost);
                let entry = match open.remove(&at) {
                    Some(old) => self.bdd.or(&old, &found)?,
                    None => found,
                };
                open.insert(at, entry);
            }
        }

        Ok(rounds)
    }

    /// `rounds` merged into one layer per price.
    fn merged(&self, rounds: &Rounds) -> BackendResult<Layers> {
        let mut layers: Layers = Vec::new();
        for (price, _, set) in rounds {
            match layers.last_mut() {
                Some((p, s)) if p == price => *s = self.bdd.or(s, set)?,
                _ => layers.push((*price, set.clone())),
            }
        }

        Ok(layers)
    }

    /// `layers` carried across a global part `relation`, cheapest price per
    /// leaf state; `None` if no leaf state survives.
    fn transfer(
        &self,
        leaf: usize,
        relation: &Set,
        layers: &Layers,
    ) -> BackendResult<Option<Layers>> {
        let mut out: Layers = Vec::new();
        let mut seen = self.bdd.constant(false);
        for (price, set) in layers {
            let moved = self.minus(&self.image(leaf, set, relation)?, &seen)?;
            if self.bdd.is_empty(&moved) {
                continue;
            }

            seen = self.bdd.or(&seen, &moved)?;
            out.push((*price, moved));
        }

        Ok((!out.is_empty()).then_some(out))
    }

    /// The least price of a leaf state in `layers` that lies in `set`.
    fn min_price(&self, layers: &Layers, set: &Set) -> BackendResult<Option<Price>> {
        for (price, layer) in layers {
            if self.meets(layer, set)? {
                return Ok(Some(*price));
            }
        }

        Ok(None)
    }

    /// The initial decoupled state.
    fn initial(&self) -> BackendResult<State> {
        let center = self.initial_center.clone();
        let leaves = (0..self.leaves.len())
            .map(|l| {
                let seeds = vec![(0, self.leaves[l].initial.clone())];
                self.merged(&self.close(l, &center, &seeds)?)
            })
            .collect::<BackendResult<_>>()?;
        Ok(State { center, leaves })
    }

    /// The successor of `state` under global action `index`; `None` if it
    /// is not applicable.
    fn successor(&self, index: usize, state: &State) -> BackendResult<Option<State>> {
        let g = &self.globals[index];
        if !g.center_pre.iter().all(|&(s, v)| state.center[s] == v) {
            return Ok(None);
        }

        let mut seeds = Vec::with_capacity(self.leaves.len());
        for (leaf, (part, layers)) in g.parts.iter().zip(&state.leaves).enumerate() {
            match part {
                None => seeds.push(layers.clone()),
                Some(relation) => match self.transfer(leaf, relation, layers)? {
                    Some(moved) => seeds.push(moved),
                    None => return Ok(None),
                },
            }
        }

        let mut center = state.center.clone();
        for &(slot, value) in &g.center_eff {
            center[slot] = value;
        }

        let leaves = seeds
            .iter()
            .enumerate()
            .map(|(l, s)| self.merged(&self.close(l, &center, s)?))
            .collect::<BackendResult<_>>()?;
        Ok(Some(State { center, leaves }))
    }

    /// The leaf cost still to pay to reach the goal from `state`; `None` if
    /// it contains no goal state.
    fn goal_price(&self, state: &State) -> BackendResult<Option<Price>> {
        if !self.center_goal.iter().all(|&(s, v)| state.center[s] == v) {
            return Ok(None);
        }

        let mut sum: Price = 0;
        for (leaf, layers) in self.leaves.iter().zip(&state.leaves) {
            match self.min_price(layers, &leaf.goal)? {
                Some(p) => sum = sum.saturating_add(p),
                None => return Ok(None),
            }
        }

        Ok(Some(sum))
    }

    /// The cheapest price of every fact over the member states of `state`.
    fn fact_prices(&self, state: &State) -> BackendResult<FactPrices<'_>> {
        let mut prices = vec![UNREACHED; *self.fact_offsets.last().expect("final entry")];
        for (var, place) in self.place.iter().enumerate() {
            let base = self.fact_offsets[var];
            match *place {
                Place::Center(slot) => prices[base + widen(state.center[slot])] = 0,
                Place::Leaf(leaf, slot) => {
                    for (value, fact) in self.leaves[leaf].facts[slot].iter().enumerate() {
                        if let Some(p) = self.min_price(&state.leaves[leaf], fact)? {
                            prices[base + value] = p;
                        }
                    }
                }
            }
        }

        Ok(FactPrices {
            offsets: Cow::Borrowed(&self.fact_offsets),
            leaves: Cow::Borrowed(&self.leaf_of),
            prices,
        })
    }

    /// Per leaf, the cheapest leaf-action cost from the initial leaf state
    /// to a goal leaf state with global parts free and center preconditions
    /// ignored, summed; [`Price::MAX`] if some leaf can never reach its goal
    /// (the bound of anytime A*).
    fn min_goal_price(&self) -> BackendResult<Price> {
        let mut sum: Price = 0;
        for (l, leaf) in self.leaves.iter().enumerate() {
            let mut moves: Vec<(Price, &Set)> =
                leaf.moves.iter().map(|m| (m.cost, &m.relation)).collect();
            moves.extend(
                self.globals
                    .iter()
                    .filter_map(|g| g.parts[l].as_ref().map(|r| (0, r))),
            );
            let mut open: BTreeMap<Price, Set> = BTreeMap::from([(0, leaf.initial.clone())]);
            let mut closed = self.bdd.constant(false);
            let mut best = None;
            while let Some((price, set)) = open.pop_first() {
                let mut frontier = self.minus(&set, &closed)?;
                let mut layer = self.bdd.constant(false);
                while !self.bdd.is_empty(&frontier) {
                    closed = self.bdd.or(&closed, &frontier)?;
                    layer = self.bdd.or(&layer, &frontier)?;
                    let mut found = self.bdd.constant(false);
                    for &(_, r) in moves.iter().filter(|(c, _)| *c == 0) {
                        found = self.bdd.or(&found, &self.image(l, &frontier, r)?)?;
                    }
                    frontier = self.minus(&found, &closed)?;
                }

                if self.meets(&layer, &leaf.goal)? {
                    best = Some(price);
                    break;
                }

                for &(cost, r) in moves.iter().filter(|(c, _)| *c > 0) {
                    let found = self.minus(&self.image(l, &layer, r)?, &closed)?;
                    if !self.bdd.is_empty(&found) {
                        let at = price.saturating_add(cost);
                        let entry = match open.remove(&at) {
                            Some(old) => self.bdd.or(&old, &found)?,
                            None => found,
                        };
                        open.insert(at, entry);
                    }
                }
            }

            sum = sum.saturating_add(best.unwrap_or(Price::MAX));
        }

        Ok(sum)
    }

    /// Whether stored `t` reached at `g_t` dominates `n` reached at `g_n`
    /// under [`Pruning::Basic`], as the explicit criterion decides it.
    fn dominates(
        &self,
        (t, g_t): (&State, Price),
        (n, g_n): (&State, Price),
        greedy: bool,
        fold_g: bool,
    ) -> BackendResult<bool> {
        let mut budget: i128 = 0;
        for (layers_n, layers_t) in n.leaves.iter().zip(&t.leaves) {
            let mut reached_t = self.bdd.constant(false);
            for (_, s) in layers_t {
                reached_t = self.bdd.or(&reached_t, s)?;
            }

            let mut delta = i128::MIN;
            for (p_n, s_n) in layers_n {
                if self.meets(s_n, &self.bdd.not(&reached_t)?)? {
                    return Ok(false);
                }

                for (p_t, s_t) in layers_t {
                    if self.meets(s_n, s_t)? {
                        delta = delta.max(i128::from(*p_t) - i128::from(*p_n));
                    }
                }
            }

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

    /// A single leaf state of `leaf` in `set`.
    fn pick(&self, leaf: usize, set: &Set) -> BackendResult<Set> {
        let bits = self.bdd.pick(set).expect("a non-empty set");
        let mut cube = self.bdd.constant(true);
        for &v in &self.leaves[leaf].encoding.cur_vars {
            let lit = self.bdd.var(v);
            let lit = if bits[v as usize] {
                lit
            } else {
                self.bdd.not(&lit)?
            };
            cube = self.bdd.and(&cube, &lit)?;
        }

        Ok(cube)
    }
}

impl Leaf {
    /// Leaf `leaf` over `vars` with its fact sets, initial state and goal;
    /// its moves are added by [`classify`].
    fn new(
        bdd: &Oxidd,
        task: &Task,
        place: &[Place],
        leaf: usize,
        vars: &[VarId],
        encoding: Encoding,
    ) -> BackendResult<Self> {
        let facts = vars
            .iter()
            .enumerate()
            .map(|(slot, &v)| {
                (0..task.variables[v].domain_size)
                    .map(|value| value_set(bdd, &encoding.bits[slot], widen(value)))
                    .collect::<BackendResult<Vec<_>>>()
            })
            .collect::<BackendResult<Vec<_>>>()?;
        let state = task.initial_state();
        let mut initial = bdd.constant(true);
        for (slot, &v) in vars.iter().enumerate() {
            initial = bdd.and(&initial, &facts[slot][widen(state.0[v])])?;
        }

        let mut goal = bdd.constant(true);
        for f in &task.goal {
            if let Place::Leaf(l, slot) = place[f.var]
                && l == leaf
            {
                goal = bdd.and(&goal, &facts[slot][widen(f.value)])?;
            }
        }

        Ok(Self {
            encoding,
            moves: Vec::new(),
            groups: Vec::new(),
            initial,
            goal,
            facts,
        })
    }

    /// Merges the moves with equal cost and center preconditions into
    /// [`Leaf::groups`].
    fn group_moves(&mut self, bdd: &Oxidd) -> BackendResult<()> {
        let mut groups: Vec<Group> = Vec::new();
        for m in &self.moves {
            match groups
                .iter_mut()
                .find(|g| g.cost == m.cost && g.center_pre == m.center_pre)
            {
                Some(g) => g.relation = bdd.or(&g.relation, &m.relation)?,
                None => groups.push(Group {
                    cost: m.cost,
                    center_pre: m.center_pre.clone(),
                    relation: m.relation.clone(),
                }),
            }
        }

        self.groups = groups;
        Ok(())
    }
}

/// Adds operator `op` to its leaf as a move if it is a leaf action (it sets
/// no center variable and touches exactly one leaf, which it sets), as the
/// explicit compilation decides; otherwise returns it as a global action.
fn classify(
    bdd: &Oxidd,
    place: &[Place],
    leaves: &mut [Leaf],
    op: OpId,
    operator: &sas_plus::Operator,
) -> BackendResult<Option<Global>> {
    let cost = Price::from(operator.cost);
    let mut eff: Vec<Fact> = Vec::new();
    for e in &operator.eff {
        eff.retain(|f| f.var != e.fact.var);
        eff.push(e.fact);
    }

    let center_pre = center_facts(place, &operator.pre);
    let center_eff = center_facts(place, &eff);
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
        let relation = relation(bdd, &leaves[leaf], place, leaf, &operator.pre, &eff)?;
        leaves[leaf].moves.push(LeafMove {
            op,
            cost,
            center_pre,
            relation,
        });
        return Ok(None);
    }

    let parts = (0..leaves.len())
        .map(|l| {
            touched
                .contains(&l)
                .then(|| relation(bdd, &leaves[l], place, l, &operator.pre, &eff))
                .transpose()
        })
        .collect::<BackendResult<Vec<_>>>()?;
    Ok(Some(Global {
        op,
        cost,
        center_pre,
        center_eff,
        parts,
    }))
}

/// A search node.
struct Node {
    /// The decoupled state.
    state: State,
    /// Cost of the global actions so far.
    g: Price,
    /// Parent node and global action index.
    parent: Option<(usize, usize)>,
}

/// Best-first search over symbolic decoupled states; the same loop as the
/// explicit search under [`Pruning::Basic`], so both expand the same
/// states in the same order.
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
            let plan = reconstruct(task, &search.path(id), cost)?;
            Some(with_true_cost(original, plan, task.reachability_only))
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
    /// Inserts the initial state and expands nodes until done.
    fn run(&mut self) -> BackendResult<()> {
        let initial = self.task.initial()?;
        self.insert(initial, 0, None)?;
        while let Some(Reverse((key, id))) = self.open.pop() {
            if self.is_done(key) {
                break;
            }

            self.statistics.expanded += 1;
            let g = self.nodes[id].g;
            let mut successors = Vec::new();
            for i in 0..self.task.globals.len() {
                if let Some(s) = self.task.successor(i, &self.nodes[id].state)? {
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
                    Some(r) => r.estimate(&self.task.fact_prices(state)?.reached_for_free()),
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
            if self
                .task
                .dominates((&node.state, node.g), (state, g), greedy, self.fold_g)?
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
            Some(h) => h.estimate(&self.task.fact_prices(&state)?),
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
        if let Some(rest) = self.task.goal_price(&state)? {
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

/// The sequential plan for a solution ending in the last state of `path`:
/// per stage the global action, then every leaf's leaf actions.
fn reconstruct(
    task: &SymbolicTask,
    path: &[(Option<usize>, &State)],
    cost: Price,
) -> BackendResult<Plan> {
    let leaf_ops = (0..task.leaves.len())
        .map(|leaf| leaf_stages(task, path, leaf))
        .collect::<BackendResult<Vec<_>>>()?;

    let mut ops = Vec::new();
    for (stage, (action, _)) in path.iter().enumerate() {
        if let Some(action) = action {
            ops.push(task.globals[*action].op);
        }

        for stages in &leaf_ops {
            ops.extend_from_slice(&stages[stage]);
        }
    }

    Ok(Plan { ops, cost })
}

/// The leaf actions of `leaf` in every stage of `path`, ending in a
/// cheapest goal leaf state of the last stage. Walks back from one leaf
/// state at a time; every step lowers `(stage, price, depth)`.
fn leaf_stages(
    task: &SymbolicTask,
    path: &[(Option<usize>, &State)],
    leaf: usize,
) -> BackendResult<Vec<Vec<OpId>>> {
    let compiled = &task.leaves[leaf];
    let last = &path.last().expect("a path has an initial state").1.leaves[leaf];
    let goal_price = task
        .min_price(last, &compiled.goal)?
        .expect("a solution reaches a leaf goal state");
    let layer = last
        .iter()
        .find_map(|(p, s)| (*p == goal_price).then_some(s))
        .expect("the goal price has a layer");
    let mut target = task.pick(leaf, &task.bdd.and(layer, &compiled.goal)?)?;

    let mut stages = vec![Vec::new(); path.len()];
    for stage in (0..path.len()).rev() {
        let (ops, seed, price) = walk_stage(task, path, leaf, stage, target)?;
        stages[stage] = ops;
        target = seed;
        if stage > 0 {
            let action = path[stage].0.expect("only the initial stage has no action");
            if let Some(relation) = &task.globals[action].parts[leaf] {
                let before = &path[stage - 1].1.leaves[leaf];
                let layer = before
                    .iter()
                    .find_map(|(p, s)| (*p == price).then_some(s))
                    .expect("a seed keeps its source price");
                let source = task
                    .bdd
                    .and(&task.preimage(leaf, &target, relation)?, layer)?;
                target = task.pick(leaf, &source)?;
            }
        }
    }

    Ok(stages)
}

/// Within stage `stage`, the leaf actions that lead from a seed to leaf
/// state `target`, in execution order, with that seed and its price.
fn walk_stage(
    task: &SymbolicTask,
    path: &[(Option<usize>, &State)],
    leaf: usize,
    stage: usize,
    mut target: Set,
) -> BackendResult<(Vec<OpId>, Set, Price)> {
    let center = &path[stage].1.center;
    let seeds = stage_seeds(task, path, leaf, stage)?;
    let rounds = task.close(leaf, center, &seeds)?;
    let key_of = |x: &Set| -> BackendResult<(Price, usize)> {
        for (p, d, s) in &rounds {
            if task.meets(s, x)? {
                return Ok((*p, *d));
            }
        }
        unreachable!("the walk stays inside the closure")
    };
    let active: Vec<&LeafMove> = task.leaves[leaf]
        .moves
        .iter()
        .filter(|m| m.center_pre.iter().all(|&(s, v)| center[s] == v))
        .collect();

    let mut ops = Vec::new();
    let (mut price, mut depth) = key_of(&target)?;
    loop {
        if depth == 0 {
            let seed = seeds.iter().find_map(|(p, s)| (*p == price).then_some(s));
            if let Some(seed) = seed
                && task.meets(seed, &target)?
            {
                break;
            }
        }

        let (op, from) = predecessor(task, leaf, &rounds, &active, &target, (price, depth))?
            .expect("every closure state has a predecessor or is a seed");
        ops.push(op);
        target = from;
        (price, depth) = key_of(&target)?;
    }

    ops.reverse();
    Ok((ops, target, price))
}

/// A move among `active` and a leaf state it leads from to `target`, which
/// has key `(price, depth)` in `rounds`: from the previous round of the same
/// price by a 0-cost move, or from any round of price `price - cost` by a
/// move of that cost.
fn predecessor(
    task: &SymbolicTask,
    leaf: usize,
    rounds: &Rounds,
    active: &[&LeafMove],
    target: &Set,
    (price, depth): (Price, usize),
) -> BackendResult<Option<(OpId, Set)>> {
    for m in active {
        let cost = task.effective(m.cost);
        let mut wanted = rounds.iter().filter(|(p, d, _)| {
            if depth > 0 {
                cost == 0 && *p == price && *d == depth - 1
            } else {
                cost > 0 && price >= cost && *p == price - cost
            }
        });
        let Some(first) = wanted.next() else {
            continue;
        };

        let before = task.preimage(leaf, target, &m.relation)?;
        for (_, _, s) in core::iter::once(first).chain(wanted) {
            let hit = task.bdd.and(&before, s)?;
            if !task.bdd.is_empty(&hit) {
                return Ok(Some((m.op, task.pick(leaf, &hit)?)));
            }
        }
    }

    Ok(None)
}

/// The seeds of `leaf` at the start of stage `stage`: the initial leaf
/// state, or the previous stage carried across the global action.
fn stage_seeds(
    task: &SymbolicTask,
    path: &[(Option<usize>, &State)],
    leaf: usize,
    stage: usize,
) -> BackendResult<Layers> {
    let Some(action) = path[stage].0 else {
        return Ok(vec![(0, task.leaves[leaf].initial.clone())]);
    };

    let before = &path[stage - 1].1.leaves[leaf];
    match &task.globals[action].parts[leaf] {
        None => Ok(before.clone()),
        Some(relation) => Ok(task
            .transfer(leaf, relation, before)?
            .expect("the path applied this action")),
    }
}

impl Encoding {
    /// The encoding of a leaf over `vars`, taking BDD variables from
    /// `next_var` on.
    fn new(bdd: &Oxidd, task: &Task, vars: &[VarId], next_var: &mut u32) -> BackendResult<Self> {
        let mut bits = Vec::new();
        let (mut cur, mut next) = (Vec::new(), Vec::new());
        for &v in vars {
            let n = width(task.variables[v].domain_size);
            let mut slot = Vec::new();
            for _ in 0..n {
                slot.push(*next_var);
                cur.push(*next_var);
                next.push(*next_var + 1);
                *next_var += 2;
            }
            bits.push(slot);
        }

        Ok(Self {
            bits,
            cur: bdd.var_set(&cur)?,
            next: bdd.var_set(&next)?,
            to_cur: bdd.renaming(&next, &cur),
            to_next: bdd.renaming(&cur, &next),
            cur_vars: cur,
        })
    }
}

/// The leaf states where the variable with `bits` has `value`, on current
/// bits (`offset` 0) or next bits (`offset` 1).
fn value_on(bdd: &Oxidd, bits: &[u32], value: usize, offset: u32) -> BackendResult<Set> {
    let mut set = bdd.constant(true);
    for (i, &b) in bits.iter().enumerate() {
        let lit = bdd.var(b + offset);
        let lit = if (value >> i) & 1 == 1 {
            lit
        } else {
            bdd.not(&lit)?
        };
        set = bdd.and(&set, &lit)?;
    }

    Ok(set)
}

/// The leaf states where the variable with `bits` has `value`.
fn value_set(bdd: &Oxidd, bits: &[u32], value: usize) -> BackendResult<Set> {
    value_on(bdd, bits, value, 0)
}

/// The relation of an operator with precondition `pre` and effects `eff`
/// on leaf `leaf`: its preconditions there, its effects on next bits, and
/// every other leaf variable unchanged.
fn relation(
    bdd: &Oxidd,
    leaf: &Leaf,
    place: &[Place],
    index: usize,
    pre: &[Fact],
    eff: &[Fact],
) -> BackendResult<Set> {
    let bits = &leaf.encoding.bits;
    let mut set = bdd.constant(true);
    for f in pre {
        if let Place::Leaf(l, slot) = place[f.var]
            && l == index
        {
            set = bdd.and(&set, &value_on(bdd, &bits[slot], widen(f.value), 0)?)?;
        }
    }

    let mut written = vec![false; bits.len()];
    for f in eff {
        if let Place::Leaf(l, slot) = place[f.var]
            && l == index
        {
            written[slot] = true;
            set = bdd.and(&set, &value_on(bdd, &bits[slot], widen(f.value), 1)?)?;
        }
    }

    for (slot, slot_bits) in bits.iter().enumerate() {
        if written[slot] {
            continue;
        }
        for &b in slot_bits {
            let same = bdd.ite(&bdd.var(b), &bdd.var(b + 1), &bdd.not(&bdd.var(b + 1))?)?;
            set = bdd.and(&set, &same)?;
        }
    }

    Ok(set)
}

/// The facts of `facts` on center variables.
fn center_facts(place: &[Place], facts: &[Fact]) -> Vec<(usize, Value)> {
    facts
        .iter()
        .filter_map(|f| match place[f.var] {
            Place::Center(slot) => Some((slot, f.value)),
            Place::Leaf(..) => None,
        })
        .collect()
}

/// Whether fact `f` lies on leaf `leaf`.
fn leaf_is(place: &[Place], f: &Fact, leaf: usize) -> bool {
    matches!(place[f.var], Place::Leaf(l, _) if l == leaf)
}
