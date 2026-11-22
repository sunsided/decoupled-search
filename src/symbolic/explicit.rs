//! Enumerated leaves for the symbolic search: leaves small enough that a
//! price per leaf state is cheaper than diagrams
//! ([`crate::Config::symbolic_min_states`]).

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{OpId, Value};

use crate::decoupled::{Price, UNREACHED};

/// A leaf's part of an operator, as `(slot, value)` lists.
pub(super) struct LocalFacts {
    /// Preconditions on the leaf.
    pub(super) pre: Vec<(usize, usize)>,
    /// Effects on the leaf.
    pub(super) eff: Vec<(usize, usize)>,
}

/// A leaf action of an enumerated leaf.
pub(super) struct Move {
    /// The operator.
    pub(super) op: OpId,
    /// Its cost.
    pub(super) cost: Price,
    /// Its center preconditions, as `(center slot, value)`.
    pub(super) center_pre: Vec<(usize, Value)>,
    /// The successor of every leaf state; `None` where it does not apply.
    pub(super) next: Vec<Option<usize>>,
}

/// An enumerated leaf.
pub(super) struct ExplicitLeaf {
    /// Rank weight of each slot.
    strides: Vec<usize>,
    /// Domain size of each slot.
    domains: Vec<usize>,
    /// Number of leaf states.
    pub(super) size: usize,
    /// Rank of the initial leaf state.
    pub(super) initial: usize,
    /// Whether each leaf state satisfies the leaf goal.
    pub(super) goal: Vec<bool>,
    /// Its leaf actions.
    pub(super) moves: Vec<Move>,
}

/// A closure result: prices and, per leaf state, the leaf state and move
/// that reached it (`None` for seeds).
pub(super) type Closure = (Vec<Price>, Vec<Option<(usize, usize)>>);

impl ExplicitLeaf {
    /// A leaf with `domains`, initial values `initial` and goal facts
    /// `goal`, both as `(slot, value)`.
    pub(super) fn new(domains: Vec<usize>, initial: &[usize], goal: &[(usize, usize)]) -> Self {
        let mut strides = Vec::with_capacity(domains.len());
        let mut size = 1;
        for &d in &domains {
            strides.push(size);
            size *= d;
        }

        let initial = initial.iter().zip(&strides).map(|(v, s)| v * s).sum();
        let mut leaf = Self {
            strides,
            domains,
            size,
            initial,
            goal: Vec::new(),
            moves: Vec::new(),
        };
        leaf.goal = (0..size).map(|x| leaf.satisfies(x, goal)).collect();
        leaf
    }

    /// The value of slot `slot` in leaf state `rank`.
    fn value(&self, rank: usize, slot: usize) -> usize {
        rank / self.strides[slot] % self.domains[slot]
    }

    /// Whether leaf state `rank` satisfies `facts`.
    pub(super) fn satisfies(&self, rank: usize, facts: &[(usize, usize)]) -> bool {
        facts.iter().all(|&(s, v)| self.value(rank, s) == v)
    }

    /// The successor table of an operator part on this leaf.
    pub(super) fn table(&self, part: &LocalFacts) -> Vec<Option<usize>> {
        (0..self.size)
            .map(|x| {
                self.satisfies(x, &part.pre).then(|| {
                    part.eff.iter().fold(x, |rank, &(s, v)| {
                        rank - self.value(rank, s) * self.strides[s] + v * self.strides[s]
                    })
                })
            })
            .collect()
    }

    /// Leaf reachability from `seeds` over the moves `active` accepts, with
    /// move costs (or 0 if `costs` is off).
    pub(super) fn close(&self, seeds: &[Price], active: super::Active<'_>, costs: bool) -> Closure {
        let moves: Vec<(usize, &Move)> = self
            .moves
            .iter()
            .enumerate()
            .filter(|(_, m)| active(&m.center_pre))
            .collect();
        let mut prices = seeds.to_vec();
        let mut parents = vec![None; self.size];
        let mut open: BinaryHeap<Reverse<(Price, usize)>> = (0..self.size)
            .filter(|&x| prices[x] != UNREACHED)
            .map(|x| Reverse((prices[x], x)))
            .collect();
        while let Some(Reverse((price, x))) = open.pop() {
            if price > prices[x] {
                continue;
            }

            for &(index, m) in &moves {
                let Some(y) = m.next[x] else { continue };
                let through = price.saturating_add(if costs { m.cost } else { 0 });
                if through < prices[y] {
                    prices[y] = through;
                    parents[y] = Some((x, index));
                    open.push(Reverse((through, y)));
                }
            }
        }

        (prices, parents)
    }

    /// `prices` carried across a global part with successor `table`;
    /// `None` if no leaf state survives.
    pub(super) fn transfer(&self, table: &[Option<usize>], prices: &[Price]) -> Option<Vec<Price>> {
        let mut next = vec![UNREACHED; self.size];
        for (x, &p) in prices.iter().enumerate() {
            if let (true, Some(y)) = (p != UNREACHED, table[x]) {
                next[y] = next[y].min(p);
            }
        }

        next.iter().any(|&p| p != UNREACHED).then_some(next)
    }

    /// A leaf state of `prices` at `price` that `table` carries to `target`.
    pub(super) fn source(
        &self,
        table: &[Option<usize>],
        prices: &[Price],
        target: usize,
        price: Price,
    ) -> usize {
        (0..self.size)
            .find(|&x| prices[x] == price && table[x] == Some(target))
            .expect("a carried leaf state has a source")
    }

    /// The least price in `prices` among the leaf states `member` accepts.
    pub(super) fn min_price(prices: &[Price], member: impl Fn(usize) -> bool) -> Option<Price> {
        (0..prices.len())
            .filter(|&x| prices[x] != UNREACHED && member(x))
            .map(|x| prices[x])
            .min()
    }

    /// The rank of a leaf state with the least price among the goal states.
    pub(super) fn cheapest_goal(&self, prices: &[Price]) -> Option<usize> {
        (0..self.size)
            .filter(|&x| self.goal[x] && prices[x] != UNREACHED)
            .min_by_key(|&x| prices[x])
    }

    /// The explicit dominance slack of stored `t` over new `n`: the largest
    /// `t(x) - n(x)` over the leaf states `n` reaches; `None` if `t` does
    /// not reach one of them.
    pub(super) fn slack(t: &[Price], n: &[Price]) -> Option<i128> {
        let mut delta = i128::MIN;
        for (x, &p) in n.iter().enumerate() {
            if p == UNREACHED {
                continue;
            }
            if t[x] == UNREACHED {
                return None;
            }
            delta = delta.max(i128::from(t[x]) - i128::from(p));
        }

        Some(delta)
    }

    /// The cheapest leaf-action cost from the initial leaf state to a goal
    /// leaf state, center preconditions ignored, global parts `free` at 0.
    pub(super) fn cheapest_goal_cost(&self, free: &[&[Option<usize>]]) -> Option<Price> {
        let mut distance = vec![UNREACHED; self.size];
        distance[self.initial] = 0;
        let mut open = BinaryHeap::from([Reverse((0, self.initial))]);
        while let Some(Reverse((d, x))) = open.pop() {
            if d > distance[x] {
                continue;
            }
            if self.goal[x] {
                return Some(d);
            }

            let leaf = self
                .moves
                .iter()
                .filter_map(|m| m.next[x].map(|y| (y, m.cost)));
            let global = free.iter().filter_map(|t| t[x].map(|y| (y, 0)));
            for (y, cost) in leaf.chain(global) {
                let through = d.saturating_add(cost);
                if through < distance[y] {
                    distance[y] = through;
                    open.push(Reverse((through, y)));
                }
            }
        }

        None
    }
}
