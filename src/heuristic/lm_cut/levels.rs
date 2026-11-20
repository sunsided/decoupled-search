//! The price-level encoding of [`super::LmCutHeuristic::with_price_levels`]:
//! Gnad and Hoffmann's `Π^L$` (ICAPS 2015, Definition 5) at the fact level,
//! rebuilt for every estimate.
//!
//! Every fact `(v, d)` of the original task is a binary variable (LM-cut
//! works on the delete relaxation, where this loses nothing):
//!
//! - facts at price 0 and the default values of derived variables (as in
//!   [`crate::HMax`]) are true initially, everything else false;
//! - every operator keeps its cost, preconditions and (conditional)
//!   effects on the binary variables; every axiom becomes a 0-cost
//!   operator;
//! - per factor (each leaf, and the center) and per positive price level
//!   `p` of its facts, one *purchase* operator of cost `p` makes every fact
//!   of that factor with price at most `p` true.
//!
//! *Admissibility.* Take any completion of the decoupled state through a
//! member state `m` that buys leaf state `x_L` at price `P_L`, followed by
//! a plan `π` from `m`. Every fact of `x_L` has a fact price of at most
//! `P_L`, so the purchase at level `P_L` (or the initial state, if
//! `P_L = 0`) makes all of them true; the derived facts of `m` follow from
//! 0-cost axiom operators or are defaults, and `π` is then applicable in
//! the delete relaxation. So the relaxed task has a plan of cost at most
//! `Σ_L P_L + cost(π)`, and `h^LM-cut <= h^+` of it. Buying more than one
//! level, or facts no member state has together, only adds options.
//! Prices above `u32::MAX` are capped, which only lowers the estimate.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use planning_heuristics::LmCut;
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

use crate::heuristic::{FactPrices, fact_offsets};

/// The price-level encoding of one task, without the per-estimate parts.
#[derive(Clone, Debug)]
pub(super) struct Levels {
    /// The fact-level encoding without its initial state and purchases.
    template: Task,
    /// Index of each original variable's first fact, plus a final entry
    /// with the number of facts.
    offsets: Vec<usize>,
    /// The facts true regardless of prices: the default values of derived
    /// variables.
    defaults: Vec<usize>,
}

impl Levels {
    /// The encoding of `task` (see the module docs).
    ///
    /// # Panics
    ///
    /// Panics if `task` has a fact value outside its variable's domain
    /// that does not fit `usize` (never for a validated task).
    pub(super) fn new(task: &Task) -> Self {
        let offsets = fact_offsets(task);
        let id = |f: &Fact| offsets[f.var] + usize::try_from(f.value).expect("value fits usize");
        let binary = |f: &Fact| Fact::new(id(f), 1);
        let facts = *offsets.last().expect("has a final entry");

        let mut operators: Vec<Operator> = task
            .operators
            .iter()
            .map(|op| Operator {
                name: op.name.clone(),
                pre: op.pre.iter().map(binary).collect(),
                eff: op
                    .eff
                    .iter()
                    .map(|e| Effect {
                        conditions: e.conditions.iter().map(binary).collect(),
                        fact: binary(&e.fact),
                    })
                    .collect(),
                cost: op.cost,
            })
            .collect();
        operators.extend(task.axioms.iter().map(|axiom| Operator {
            name: "axiom".into(),
            pre: axiom.conditions.iter().map(binary).collect(),
            eff: vec![Effect::new(binary(&axiom.head))],
            cost: 0,
        }));

        let defaults = (0..task.num_variables())
            .filter(|&v| task.variables[v].is_derived())
            .map(|v| id(&Fact::new(v, task.initial.0[v])))
            .collect();
        let template = Task {
            variables: (0..facts)
                .map(|f| Variable::new(format!("f{f}"), 2))
                .collect(),
            operators,
            initial: State(vec![0; facts]),
            goal: task.goal.iter().map(binary).collect(),
            axioms: vec![],
        };

        Self {
            template,
            offsets,
            defaults,
        }
    }

    /// The encoding seeded with `facts`: initial state and purchases.
    fn seeded(&self, facts: &FactPrices<'_>) -> Task {
        let mut task = self.template.clone();
        for &f in &self.defaults {
            task.initial.0[f] = 1;
        }

        // Per factor (`None` is the center): its priced facts.
        let mut priced: Vec<(Option<usize>, u64, usize)> = Vec::new();
        for var in 0..self.offsets.len() - 1 {
            for f in self.offsets[var]..self.offsets[var + 1] {
                match facts.prices[f] {
                    0 => task.initial.0[f] = 1,
                    u64::MAX => {}
                    price => priced.push((facts.leaf_of(var), price, f)),
                }
            }
        }

        priced.sort_unstable();
        let mut start = 0;
        while start < priced.len() {
            let (factor, price, _) = priced[start];
            let mut end = start;
            while end < priced.len() && priced[end].0 == factor && priced[end].1 == price {
                end += 1;
            }

            let first = priced.partition_point(|&(other, ..)| other < factor);
            task.operators.push(purchase(&priced[first..end], price));
            start = end;
        }

        task
    }
}

/// The purchase of every fact in `facts` (one factor's facts up to and
/// including price level `price`) at cost `price`, capped to `u32::MAX`.
fn purchase(facts: &[(Option<usize>, u64, usize)], price: u64) -> Operator {
    Operator {
        name: "purchase".into(),
        pre: vec![],
        eff: facts
            .iter()
            .map(|&(_, _, f)| Effect::new(Fact::new(f, 1)))
            .collect(),
        cost: u32::try_from(price).unwrap_or(u32::MAX),
    }
}

impl Levels {
    /// `h^LM-cut` of the encoding seeded with `facts`; `None` for a dead
    /// end. Builds the task and a fresh `LmCut` every time.
    pub(super) fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64> {
        let task = self.seeded(facts);
        LmCut::new(&task).value(&task.initial)
    }
}
