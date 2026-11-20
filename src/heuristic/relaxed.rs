//! Delete-relaxed exploration of a task from priced seed facts.

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::{Fact, Task};

use super::FactPrices;

/// The cost of a fact that the relaxation does not reach.
const UNREACHED: u64 = u64::MAX;

/// How an operator combines the costs of its preconditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combine {
    /// The largest precondition cost (`h^max`).
    Max,
    /// The sum of the precondition costs (`h^add`).
    Sum,
}

/// An operator of the delete relaxation.
#[derive(Clone, Debug)]
struct RelaxedOp {
    /// Fact indices of the preconditions, without duplicates.
    pre: Vec<usize>,
    /// Fact indices of the effects.
    eff: Vec<usize>,
    /// The operator cost.
    cost: u64,
}

/// The cost of every fact after an exploration, and the operator that last
/// lowered it (`None` for facts at their seed price).
struct Exploration {
    /// Cost of every fact, [`UNREACHED`] if not reached.
    cost: Vec<u64>,
    /// The best supporter of every fact.
    supporter: Vec<Option<usize>>,
}

/// A task compiled for delete-relaxed exploration over fact indices.
#[derive(Clone, Debug)]
pub(crate) struct Relaxation {
    /// Every fact of the task, by fact index.
    facts: Vec<Fact>,
    /// The relaxed operators, by operator id.
    ops: Vec<RelaxedOp>,
    /// For every fact, the operators that have it as a precondition.
    pre_of: Vec<Vec<usize>>,
    /// Fact indices of the goal.
    goal: Vec<usize>,
}

impl Relaxation {
    /// Compiles `task`.
    ///
    /// # Panics
    ///
    /// Panics if `task` fails [`Task::validate`].
    pub(crate) fn new(task: &Task) -> Self {
        let mut offsets = Vec::with_capacity(task.num_variables());
        let mut facts = Vec::new();
        for (var, variable) in task.variables.iter().enumerate() {
            offsets.push(facts.len());
            facts.extend((0..variable.domain_size).map(|value| Fact::new(var, value)));
        }

        let index = |f: &Fact| offsets[f.var] + usize::try_from(f.value).expect("u32 fits usize");
        let ops: Vec<RelaxedOp> = task
            .operators
            .iter()
            .map(|op| {
                let mut pre: Vec<usize> = op.pre.iter().map(index).collect();
                pre.sort_unstable();
                pre.dedup();
                RelaxedOp {
                    pre,
                    eff: op.eff.iter().map(|e| index(&e.fact)).collect(),
                    cost: u64::from(op.cost),
                }
            })
            .collect();

        let mut pre_of = vec![Vec::new(); facts.len()];
        for (id, op) in ops.iter().enumerate() {
            for &f in &op.pre {
                pre_of[f].push(id);
            }
        }

        Self {
            facts,
            ops,
            pre_of,
            goal: task.goal.iter().map(index).collect(),
        }
    }

    /// `h^max` from the seed `prices`; `None` if a goal fact is unreachable.
    pub(crate) fn h_max(&self, prices: &FactPrices<'_>) -> Option<u64> {
        let explored = self.explore(prices, Combine::Max);
        self.goal.iter().try_fold(0, |h, &g| {
            let cost = explored.cost[g];
            (cost != UNREACHED).then_some(h.max(cost))
        })
    }

    /// `h^FF` from the seed `prices`: the cost of the relaxed plan from
    /// `h^add` best supporters plus the largest seed price of a fact it
    /// starts from; `None` if a goal fact is unreachable.
    pub(crate) fn h_ff(&self, prices: &FactPrices<'_>) -> Option<u64> {
        let explored = self.explore(prices, Combine::Sum);
        if self.goal.iter().any(|&g| explored.cost[g] == UNREACHED) {
            return None;
        }

        let mut seen = vec![false; self.facts.len()];
        let mut used = vec![false; self.ops.len()];
        let mut stack = self.goal.clone();
        let (mut plan_cost, mut seed_cost) = (0_u64, 0_u64);
        while let Some(f) = stack.pop() {
            if core::mem::replace(&mut seen[f], true) {
                continue;
            }

            match explored.supporter[f] {
                Some(op) if !core::mem::replace(&mut used[op], true) => {
                    plan_cost = plan_cost.saturating_add(self.ops[op].cost);
                    stack.extend_from_slice(&self.ops[op].pre);
                }
                Some(_) => {}
                None => seed_cost = seed_cost.max(explored.cost[f]),
            }
        }

        Some(plan_cost.saturating_add(seed_cost))
    }

    /// Generalised Dijkstra over facts from the seed `prices`, combining
    /// precondition costs with `combine`.
    ///
    /// Supporters change only on strict improvement and point to operators
    /// whose preconditions were settled earlier, so they never form cycles.
    fn explore(&self, prices: &FactPrices<'_>, combine: Combine) -> Exploration {
        let mut run = Run::seeded(self, prices);
        for op in (0..self.ops.len()).filter(|&op| self.ops[op].pre.is_empty()) {
            run.fire(self, op);
        }

        let mut settled = vec![false; self.facts.len()];
        while let Some(Reverse((c, f))) = run.open.pop() {
            if c > run.explored.cost[f] || core::mem::replace(&mut settled[f], true) {
                continue;
            }

            for &op in &self.pre_of[f] {
                run.value[op] = match combine {
                    Combine::Max => run.value[op].max(c),
                    Combine::Sum => run.value[op].saturating_add(c),
                };
                run.missing[op] -= 1;
                if run.missing[op] == 0 {
                    run.fire(self, op);
                }
            }
        }

        run.explored
    }
}

/// The working state of one [`Relaxation::explore`] call.
struct Run {
    /// Fact costs and supporters found so far.
    explored: Exploration,
    /// Facts by tentative cost.
    open: BinaryHeap<Reverse<(u64, usize)>>,
    /// Per operator, the number of preconditions not yet settled.
    missing: Vec<usize>,
    /// Per operator, the combined cost of its settled preconditions.
    value: Vec<u64>,
}

impl Run {
    /// Starts an exploration of `relaxation` with every priced fact open.
    fn seeded(relaxation: &Relaxation, prices: &FactPrices<'_>) -> Self {
        let cost: Vec<u64> = relaxation
            .facts
            .iter()
            .map(|&f| prices.get(f).unwrap_or(UNREACHED))
            .collect();
        let open = cost
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c != UNREACHED)
            .map(|(f, &c)| Reverse((c, f)))
            .collect();

        Self {
            explored: Exploration {
                supporter: vec![None; cost.len()],
                cost,
            },
            open,
            missing: relaxation.ops.iter().map(|op| op.pre.len()).collect(),
            value: vec![0; relaxation.ops.len()],
        }
    }

    /// Applies operator `op`, whose preconditions are all settled, lowering
    /// the cost of its effects.
    fn fire(&mut self, relaxation: &Relaxation, op: usize) {
        let reached = self.value[op].saturating_add(relaxation.ops[op].cost);
        for &e in &relaxation.ops[op].eff {
            if reached < self.explored.cost[e] {
                self.explored.cost[e] = reached;
                self.explored.supporter[e] = Some(op);
                self.open.push(Reverse((reached, e)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::borrow::Cow;
    use alloc::vec;

    use sas_plus::{Effect, Operator, State, Variable};

    use super::*;

    /// Three binary variables; from `a = 0`, operator 0 sets `b` (cost 2)
    /// and operator 1 sets `c` (cost 3). The goal is `b = 1, c = 1`.
    fn parallel() -> Task {
        let op = |eff: Fact, cost| Operator {
            name: "op".into(),
            pre: vec![Fact::new(0, 0)],
            eff: vec![Effect::new(eff)],
            cost,
        };
        Task {
            variables: vec![
                Variable::new("a", 2),
                Variable::new("b", 2),
                Variable::new("c", 2),
            ],
            operators: vec![op(Fact::new(1, 1), 2), op(Fact::new(2, 1), 3)],
            initial: State(vec![0, 0, 0]),
            goal: vec![Fact::new(1, 1), Fact::new(2, 1)],
            axioms: vec![],
        }
    }

    /// Fact prices over the three binary variables with only `a = 0` priced.
    fn seeded(offsets: &[usize], price: u64) -> FactPrices<'_> {
        let mut prices = vec![u64::MAX; 6];
        prices[0] = price;
        FactPrices {
            offsets: Cow::Borrowed(offsets),
            leaves: Cow::Owned(vec![None; 3]),
            prices,
        }
    }

    /// `h^max` takes the costlier goal, `h^FF` both supporters; seed prices
    /// add to both.
    #[test]
    fn hand_checked_values() {
        let relaxation = Relaxation::new(&parallel());
        let offsets = [0, 2, 4, 6];

        assert_eq!(relaxation.h_max(&seeded(&offsets, 0)), Some(3));
        assert_eq!(relaxation.h_ff(&seeded(&offsets, 0)), Some(5));
        assert_eq!(relaxation.h_max(&seeded(&offsets, 4)), Some(7));
        assert_eq!(relaxation.h_ff(&seeded(&offsets, 4)), Some(9));
        assert_eq!(relaxation.h_max(&seeded(&offsets, u64::MAX)), None);
        assert_eq!(relaxation.h_ff(&seeded(&offsets, u64::MAX)), None);
    }
}
