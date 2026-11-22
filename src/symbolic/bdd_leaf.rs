//! Leaves whose pricing function is kept as BDD layers, one per price
//! (Gnad, Torralba and Hoffmann 2017).

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{OpId, Value};
use symbolic_search::backend::{Backend, BackendResult, Oxidd, OxiddRenaming, OxiddVarSet};

use super::{Active, Set, holds};
use crate::decoupled::Price;

/// The pricing function of one leaf: `(price, leaf states)`, ascending in
/// price, pairwise disjoint, never empty, each state at its cheapest price.
pub(super) type Layers = Vec<(Price, Set)>;

/// A leaf closure with the 0-cost rounds kept apart: `(price, depth,
/// leaf states)` in the order found, for plan reconstruction.
type Rounds = Vec<(Price, usize, Set)>;

/// The binary encoding of a factor: interleaved current and next bits.
pub(super) struct Encoding {
    /// Current-state BDD variables of each variable, least significant bit
    /// first.
    pub(super) bits: Vec<Vec<u32>>,
    /// All current-state variables.
    cur: OxiddVarSet,
    /// All next-state variables.
    next: OxiddVarSet,
    /// Next to current.
    to_cur: OxiddRenaming,
    /// Current to next.
    to_next: OxiddRenaming,
    /// Every current-state variable, for singleton cubes.
    cur_vars: Vec<u32>,
}

impl Encoding {
    /// An encoding with `widths` bits per variable, taking BDD variables
    /// from `next_var` on.
    pub(super) fn new(bdd: &Oxidd, widths: &[usize], next_var: &mut u32) -> BackendResult<Self> {
        let mut bits = Vec::new();
        let (mut cur, mut next) = (Vec::new(), Vec::new());
        for &n in widths {
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

    /// The states where slot `slot` has `value`, on current bits (`offset`
    /// 0) or next bits (`offset` 1).
    pub(super) fn value(
        &self,
        bdd: &Oxidd,
        slot: usize,
        value: usize,
        offset: u32,
    ) -> BackendResult<Set> {
        let mut set = bdd.constant(true);
        for (i, &b) in self.bits[slot].iter().enumerate() {
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

    /// The relation that requires `pre`, sets `eff` (both `(slot, value)`)
    /// and keeps every other slot.
    pub(super) fn relation(
        &self,
        bdd: &Oxidd,
        pre: &[(usize, usize)],
        eff: &[(usize, usize)],
    ) -> BackendResult<Set> {
        let mut set = bdd.constant(true);
        for &(slot, value) in pre {
            set = bdd.and(&set, &self.value(bdd, slot, value, 0)?)?;
        }

        let mut written = vec![false; self.bits.len()];
        for &(slot, value) in eff {
            written[slot] = true;
            set = bdd.and(&set, &self.value(bdd, slot, value, 1)?)?;
        }

        for (slot, bits) in self.bits.iter().enumerate() {
            if written[slot] {
                continue;
            }
            for &b in bits {
                let same = bdd.ite(&bdd.var(b), &bdd.var(b + 1), &bdd.not(&bdd.var(b + 1))?)?;
                set = bdd.and(&set, &same)?;
            }
        }

        Ok(set)
    }

    /// The image of `set` under `relation`.
    pub(super) fn image(&self, bdd: &Oxidd, set: &Set, relation: &Set) -> BackendResult<Set> {
        let next = bdd.and_exists(set, relation, &self.cur)?;
        bdd.rename(&next, &self.to_cur)
    }

    /// The preimage of `set` under `relation`.
    pub(super) fn preimage(&self, bdd: &Oxidd, set: &Set, relation: &Set) -> BackendResult<Set> {
        let next = bdd.rename(set, &self.to_next)?;
        bdd.and_exists(&next, relation, &self.next)
    }

    /// One member of the non-empty `set`, as a cube over this encoding.
    pub(super) fn pick(&self, bdd: &Oxidd, set: &Set) -> BackendResult<Set> {
        let bits = bdd.pick(set).expect("a non-empty set");
        let mut cube = bdd.constant(true);
        for &v in &self.cur_vars {
            let lit = bdd.var(v);
            let lit = if bits[v as usize] {
                lit
            } else {
                bdd.not(&lit)?
            };
            cube = bdd.and(&cube, &lit)?;
        }

        Ok(cube)
    }

    /// The value of every slot in the single member of `cube`.
    pub(super) fn decode(&self, bdd: &Oxidd, cube: &Set) -> Vec<usize> {
        let bits = bdd.pick(cube).expect("a non-empty set");
        self.bits
            .iter()
            .map(|slot| {
                slot.iter()
                    .enumerate()
                    .map(|(i, &b)| usize::from(bits[b as usize]) << i)
                    .sum()
            })
            .collect()
    }
}

/// `a \ b`.
pub(super) fn minus(bdd: &Oxidd, a: &Set, b: &Set) -> BackendResult<Set> {
    let not_b = bdd.not(b)?;
    bdd.and(a, &not_b)
}

/// Whether `a` and `b` share a member.
pub(super) fn meets(bdd: &Oxidd, a: &Set, b: &Set) -> BackendResult<bool> {
    Ok(!bdd.is_empty(&bdd.and(a, b)?))
}

/// A leaf action on its leaf.
pub(super) struct LeafMove {
    /// The operator.
    pub(super) op: OpId,
    /// Its cost.
    pub(super) cost: Price,
    /// Its center preconditions, as `(center slot, value)`.
    pub(super) center_pre: Vec<(usize, Value)>,
    /// Its transition relation on the leaf.
    pub(super) relation: Set,
}

/// The leaf actions with equal cost and center preconditions, merged.
struct Group {
    /// The shared cost.
    cost: Price,
    /// The shared center preconditions.
    center_pre: Vec<(usize, Value)>,
    /// The union of the members' relations.
    relation: Set,
}

/// A leaf with symbolic pricing.
pub(super) struct SymLeaf {
    /// The encoding.
    pub(super) encoding: Encoding,
    /// Its leaf actions.
    pub(super) moves: Vec<LeafMove>,
    /// Its leaf actions grouped for closure.
    groups: Vec<Group>,
    /// The initial leaf state.
    pub(super) initial: Set,
    /// The leaf states satisfying the leaf goal (all if it has none).
    pub(super) goal: Set,
    /// The leaf states satisfying each fact, by slot and value.
    pub(super) facts: Vec<Vec<Set>>,
}

impl SymLeaf {
    /// A leaf with `encoding` over variables of `domains`, initial values
    /// `initial` and goal facts `goal` (`(slot, value)`).
    pub(super) fn new(
        bdd: &Oxidd,
        encoding: Encoding,
        domains: &[usize],
        initial: &[usize],
        goal: &[(usize, usize)],
    ) -> BackendResult<Self> {
        let facts = domains
            .iter()
            .enumerate()
            .map(|(slot, &d)| {
                (0..d)
                    .map(|v| encoding.value(bdd, slot, v, 0))
                    .collect::<BackendResult<Vec<_>>>()
            })
            .collect::<BackendResult<Vec<_>>>()?;
        let mut init = bdd.constant(true);
        for (slot, &v) in initial.iter().enumerate() {
            init = bdd.and(&init, &facts[slot][v])?;
        }
        let mut goal_set = bdd.constant(true);
        for &(slot, v) in goal {
            goal_set = bdd.and(&goal_set, &facts[slot][v])?;
        }

        Ok(Self {
            encoding,
            moves: Vec::new(),
            groups: Vec::new(),
            initial: init,
            goal: goal_set,
            facts,
        })
    }

    /// Merges the moves with equal cost and center preconditions.
    pub(super) fn group_moves(&mut self, bdd: &Oxidd) -> BackendResult<()> {
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

    /// The center preconditions the closure depends on.
    pub(super) fn conditions(&self) -> impl Iterator<Item = &[(usize, Value)]> {
        self.groups.iter().map(|g| g.center_pre.as_slice())
    }

    /// Symbolic Dijkstra from `seeds` over the groups `active` accepts,
    /// with costs (or 0 if `costs` is off), 0-cost rounds kept apart.
    fn rounds(
        &self,
        bdd: &Oxidd,
        seeds: &Layers,
        active: Active<'_>,
        costs: bool,
    ) -> BackendResult<Rounds> {
        let cost = |g: &Group| if costs { g.cost } else { 0 };
        let groups: Vec<&Group> = self
            .groups
            .iter()
            .filter(|g| active(&g.center_pre))
            .collect();
        let mut open: BTreeMap<Price, Set> = seeds.iter().cloned().collect();
        let mut closed = bdd.constant(false);
        let mut rounds = Vec::new();
        while let Some((price, set)) = open.pop_first() {
            let mut frontier = minus(bdd, &set, &closed)?;
            let mut layer = bdd.constant(false);
            let mut depth = 0;
            while !bdd.is_empty(&frontier) {
                closed = bdd.or(&closed, &frontier)?;
                layer = bdd.or(&layer, &frontier)?;
                rounds.push((price, depth, frontier.clone()));
                let mut found = bdd.constant(false);
                for g in groups.iter().filter(|g| cost(g) == 0) {
                    found = bdd.or(&found, &self.encoding.image(bdd, &frontier, &g.relation)?)?;
                }
                frontier = minus(bdd, &found, &closed)?;
                depth += 1;
            }

            if bdd.is_empty(&layer) {
                continue;
            }

            for g in groups.iter().filter(|g| cost(g) > 0) {
                let found = minus(
                    bdd,
                    &self.encoding.image(bdd, &layer, &g.relation)?,
                    &closed,
                )?;
                if bdd.is_empty(&found) {
                    continue;
                }

                let at = price.saturating_add(cost(g));
                let entry = match open.remove(&at) {
                    Some(old) => bdd.or(&old, &found)?,
                    None => found,
                };
                open.insert(at, entry);
            }
        }

        Ok(rounds)
    }

    /// Leaf reachability from `seeds`, one layer per price.
    pub(super) fn close(
        &self,
        bdd: &Oxidd,
        seeds: &Layers,
        active: Active<'_>,
        costs: bool,
    ) -> BackendResult<Layers> {
        let mut layers: Layers = Vec::new();
        for (price, _, set) in self.rounds(bdd, seeds, active, costs)? {
            match layers.last_mut() {
                Some((p, s)) if *p == price => *s = bdd.or(s, &set)?,
                _ => layers.push((price, set)),
            }
        }

        Ok(layers)
    }

    /// `layers` carried across a global part `relation`, cheapest price per
    /// leaf state; `None` if no leaf state survives.
    pub(super) fn transfer(
        &self,
        bdd: &Oxidd,
        relation: &Set,
        layers: &Layers,
    ) -> BackendResult<Option<Layers>> {
        let mut out: Layers = Vec::new();
        let mut seen = bdd.constant(false);
        for (price, set) in layers {
            let moved = minus(bdd, &self.encoding.image(bdd, set, relation)?, &seen)?;
            if bdd.is_empty(&moved) {
                continue;
            }

            seen = bdd.or(&seen, &moved)?;
            out.push((*price, moved));
        }

        Ok((!out.is_empty()).then_some(out))
    }

    /// The least price of a leaf state in `layers` that lies in `set`.
    pub(super) fn min_price(
        bdd: &Oxidd,
        layers: &Layers,
        set: &Set,
    ) -> BackendResult<Option<Price>> {
        for (price, layer) in layers {
            if meets(bdd, layer, set)? {
                return Ok(Some(*price));
            }
        }

        Ok(None)
    }

    /// The largest `t(x) - n(x)` over the leaf states `n` reaches; `None`
    /// if `t` does not reach one of them.
    pub(super) fn slack(bdd: &Oxidd, t: &Layers, n: &Layers) -> BackendResult<Option<i128>> {
        let mut reached_t = bdd.constant(false);
        for (_, s) in t {
            reached_t = bdd.or(&reached_t, s)?;
        }
        let outside = bdd.not(&reached_t)?;

        let mut delta = i128::MIN;
        for (p_n, s_n) in n {
            if meets(bdd, s_n, &outside)? {
                return Ok(None);
            }

            for (p_t, s_t) in t {
                if meets(bdd, s_n, s_t)? {
                    delta = delta.max(i128::from(*p_t) - i128::from(*p_n));
                }
            }
        }

        Ok(Some(delta))
    }

    /// The cheapest leaf-action cost from the initial leaf state to a goal
    /// leaf state, center preconditions ignored, global parts `free` at 0.
    pub(super) fn cheapest_goal_cost(
        &self,
        bdd: &Oxidd,
        free: &[&Set],
    ) -> BackendResult<Option<Price>> {
        let mut moves: Vec<(Price, &Set)> =
            self.moves.iter().map(|m| (m.cost, &m.relation)).collect();
        moves.extend(free.iter().map(|r| (0, *r)));
        let mut open: BTreeMap<Price, Set> = BTreeMap::from([(0, self.initial.clone())]);
        let mut closed = bdd.constant(false);
        while let Some((price, set)) = open.pop_first() {
            let mut frontier = minus(bdd, &set, &closed)?;
            let mut layer = bdd.constant(false);
            while !bdd.is_empty(&frontier) {
                closed = bdd.or(&closed, &frontier)?;
                layer = bdd.or(&layer, &frontier)?;
                let mut found = bdd.constant(false);
                for &(_, r) in moves.iter().filter(|(c, _)| *c == 0) {
                    found = bdd.or(&found, &self.encoding.image(bdd, &frontier, r)?)?;
                }
                frontier = minus(bdd, &found, &closed)?;
            }

            if meets(bdd, &layer, &self.goal)? {
                return Ok(Some(price));
            }

            for &(cost, r) in moves.iter().filter(|(c, _)| *c > 0) {
                let found = minus(bdd, &self.encoding.image(bdd, &layer, r)?, &closed)?;
                if !bdd.is_empty(&found) {
                    let at = price.saturating_add(cost);
                    let entry = match open.remove(&at) {
                        Some(old) => bdd.or(&old, &found)?,
                        None => found,
                    };
                    open.insert(at, entry);
                }
            }
        }

        Ok(None)
    }

    /// The leaf actions of every stage of a solution path ending in a
    /// cheapest goal leaf state of the last stage. Stage `k` has concrete
    /// center `centers[k]`, pricing `pricings[k]` and, for `k > 0`, the
    /// part `parts[k]` of the global action into it on this leaf. Walks
    /// back one leaf state at a time; each step lowers `(stage, price,
    /// depth)`.
    pub(super) fn stages(
        &self,
        bdd: &Oxidd,
        costs: bool,
        centers: &[Vec<Value>],
        parts: &[Option<&Set>],
        pricings: &[&Layers],
    ) -> BackendResult<Vec<Vec<OpId>>> {
        let last = pricings.last().expect("a path has an initial state");
        let goal_price =
            Self::min_price(bdd, last, &self.goal)?.expect("a solution reaches a leaf goal state");
        let layer = last
            .iter()
            .find_map(|(p, s)| (*p == goal_price).then_some(s))
            .expect("the goal price has a layer");
        let mut target = self.encoding.pick(bdd, &bdd.and(layer, &self.goal)?)?;

        let mut stages = vec![Vec::new(); centers.len()];
        for stage in (0..centers.len()).rev() {
            let seeds = match (stage, parts[stage]) {
                (0, _) => vec![(0, self.initial.clone())],
                (_, None) => pricings[stage - 1].clone(),
                (_, Some(r)) => self
                    .transfer(bdd, r, pricings[stage - 1])?
                    .expect("the path applied this action"),
            };
            let center = &centers[stage];
            let active = |pre: &[(usize, Value)]| holds(pre, center);
            let (ops, seed, price) = self.walk(bdd, costs, &seeds, &active, target)?;
            stages[stage] = ops;
            target = seed;
            if let (true, Some(relation)) = (stage > 0, parts[stage]) {
                let layer = pricings[stage - 1]
                    .iter()
                    .find_map(|(p, s)| (*p == price).then_some(s))
                    .expect("a seed keeps its source price");
                let source = bdd.and(&self.encoding.preimage(bdd, &target, relation)?, layer)?;
                target = self.encoding.pick(bdd, &source)?;
            }
        }

        Ok(stages)
    }

    /// Within one stage, the leaf actions from a seed to `target` in
    /// execution order, with that seed and its price.
    fn walk(
        &self,
        bdd: &Oxidd,
        costs: bool,
        seeds: &Layers,
        active: Active<'_>,
        mut target: Set,
    ) -> BackendResult<(Vec<OpId>, Set, Price)> {
        let rounds = self.rounds(bdd, seeds, active, costs)?;
        let key_of = |x: &Set| -> BackendResult<(Price, usize)> {
            for (p, d, s) in &rounds {
                if meets(bdd, s, x)? {
                    return Ok((*p, *d));
                }
            }
            unreachable!("the walk stays inside the closure")
        };
        let moves: Vec<&LeafMove> = self
            .moves
            .iter()
            .filter(|m| active(&m.center_pre))
            .collect();

        let mut ops = Vec::new();
        let (mut price, mut depth) = key_of(&target)?;
        loop {
            if depth == 0
                && let Some(seed) = seeds.iter().find_map(|(p, s)| (*p == price).then_some(s))
                && meets(bdd, seed, &target)?
            {
                break;
            }

            let (op, from) = self
                .predecessor(bdd, costs, &rounds, &moves, &target, (price, depth))?
                .expect("every closure state has a predecessor or is a seed");
            ops.push(op);
            target = from;
            (price, depth) = key_of(&target)?;
        }

        ops.reverse();
        Ok((ops, target, price))
    }

    /// A move and a leaf state it leads from to `target` at key `(price,
    /// depth)`: from the previous round of the same price by a 0-cost move,
    /// or from any round of price `price - cost` by a move of that cost.
    fn predecessor(
        &self,
        bdd: &Oxidd,
        costs: bool,
        rounds: &Rounds,
        moves: &[&LeafMove],
        target: &Set,
        (price, depth): (Price, usize),
    ) -> BackendResult<Option<(OpId, Set)>> {
        for m in moves {
            let cost = if costs { m.cost } else { 0 };
            let before = self.encoding.preimage(bdd, target, &m.relation)?;
            for (p, d, s) in rounds {
                let wanted = if depth > 0 {
                    cost == 0 && *p == price && *d == depth - 1
                } else {
                    cost > 0 && price >= cost && *p == price - cost
                };
                if !wanted {
                    continue;
                }

                let hit = bdd.and(&before, s)?;
                if !bdd.is_empty(&hit) {
                    return Ok(Some((m.op, self.encoding.pick(bdd, &hit)?)));
                }
            }
        }

        Ok(None)
    }
}
