//! The landmark-cut heuristic on decoupled states, from the
//! [`planning_heuristics`] crate (feature `lm-cut`). See [`LmCutHeuristic`]
//! for the two evaluations and their admissibility proofs.

mod levels;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Reverse;

use planning_heuristics::LmCut;
use sas_plus::{Fact, Task, VarId};

use super::{DecoupledHeuristic, FactPrices, HMax, fact_offsets};
use levels::Levels;

/// `h^LM-cut` (Helmert and Domshlak 2009) on decoupled states, from
/// [`planning_heuristics::LmCut`].
///
/// # Seeded evaluation ([`LmCutHeuristic::new`])
///
/// `LmCut` is built once on the task. Each estimate seeds every fact with
/// a finite price with a *share* of it ([`LmCut::value_seeded`]: a seeded
/// fact is available at the start for its seed, and seeds add up over the
/// facts a plan uses), and returns the larger of that and [`HMax`]:
///
/// - in every factor (each leaf, and the center), one *representative*
///   variable keeps the full price of its facts, every other variable's
///   facts are seeded at 0. The representative is the variable with the
///   most expensive goal fact, else the most expensive fact, the smallest
///   variable on ties; any choice is sound;
/// - facts without a price are not seeded.
///
/// *Admissibility.* `value_seeded` never exceeds the minimum, over full
/// states `s` with every fact seeded, of `Σ_v seed(s[v]) + h*(s)`. A member
/// state `m` of the decoupled state buys leaf state `x_L` at price `P_L`;
/// each fact of `x_L` (attached derived ones included) is priced at most
/// `P_L`, so it is seeded, and the shares of `x_L`'s facts sum to the price
/// of its representative fact, at most `P_L`. Center facts of `m` are
/// priced 0. So `m` is such a state with `Σ_v seed(m[v]) <= Σ_L P_L`, and the
/// estimate is at most `Σ_L P_L + h*(m)`, the cost of the cheapest
/// completion through `m`. [`HMax`] is admissible on its own, so is the
/// maximum. Seeding every fact at its full price instead would pay a leaf
/// state's price once per variable and could overestimate.
///
/// For single-variable leaves the shares are the prices themselves. For
/// leaves with several variables they keep only the representative's part;
/// the `h^max` term keeps the joint price (it charges a leaf state once).
///
/// # Price-level evaluation ([`LmCutHeuristic::with_price_levels`])
///
/// Gnad and Hoffmann's `Π^L$` (ICAPS 2015, Definition 5) at the fact level,
/// rebuilt for every estimate: one binary variable per fact, and per factor
/// and price level `p` an operator of cost `p` that buys every fact of the
/// factor priced at most `p`. It charges a multi-variable leaf state's
/// price once for all its facts, but builds a task and an `LmCut` per
/// estimate. Its proof is in the `levels` module docs.
///
/// Both are admissible; neither dominates the other in general.
///
/// # Examples
///
/// ```
/// use decoupled_search::{Factoring, LmCutHeuristic, astar};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // Two goals behind one switch: h^max sees one of them, LM-cut both.
/// let set = |var| Operator {
///     name: "set".into(),
///     pre: vec![],
///     eff: vec![Effect::new(Fact::new(var, 1))],
///     cost: 2,
/// };
/// let task = Task {
///     variables: vec![Variable::new("a", 2), Variable::new("b", 2), Variable::new("c", 2)],
///     operators: vec![
///         set(0),
///         Operator { pre: vec![Fact::new(0, 1)], ..set(1) },
///         Operator { pre: vec![Fact::new(0, 1)], ..set(2) },
///     ],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![Fact::new(1, 1), Fact::new(2, 1)],
///     axioms: vec![],
/// };
///
/// let factoring = Factoring::fork(&task);
/// for h in [LmCutHeuristic::new(&task), LmCutHeuristic::with_price_levels(&task)] {
///     assert_eq!(astar(&task, &factoring, &h)?.map(|p| p.cost), Some(6));
/// }
/// # Ok::<(), decoupled_search::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct LmCutHeuristic(Mode);

/// How [`LmCutHeuristic`] evaluates.
#[derive(Clone, Debug)]
enum Mode {
    /// Shares of the prices as seeds of one prebuilt `LmCut`.
    Seeded(Box<Seeded>),
    /// The price-level encoding, rebuilt per estimate.
    Levels(Levels),
}

/// How a variable ranks as its factor's representative, greater is better:
/// its goal fact's price, its most expensive priced fact, the smaller
/// variable.
type RepresentativeKey = (u64, u64, Reverse<VarId>);

/// The prebuilt state of the seeded evaluation.
#[derive(Clone, Debug)]
struct Seeded {
    /// `LmCut` on the task and the seed buffer, reused across estimates
    /// (`LmCut` needs `&mut` to evaluate).
    lm_cut: RefCell<(LmCut<'static>, Vec<Option<u64>>)>,
    /// `h^max` on fact prices, the other half of the maximum.
    h_max: HMax,
    /// Index of each variable's first fact in [`FactPrices`], plus a final
    /// entry with the number of facts.
    offsets: Vec<usize>,
    /// Per fact (by [`Seeded::offsets`]), its id in the `LmCut` relaxation.
    ids: Vec<usize>,
    /// Per variable, the goal value, if the goal has one.
    goal: Vec<Option<usize>>,
}

impl LmCutHeuristic {
    /// The seeded evaluation on `task` (see [`LmCutHeuristic`]): `LmCut` is
    /// built once, each estimate only seeds it.
    ///
    /// # Panics
    ///
    /// Panics if a goal fact's value does not fit `usize`.
    #[must_use]
    pub fn new(task: &Task) -> Self {
        let lm_cut = LmCut::new(task);
        let offsets = fact_offsets(task);
        let ids = (0..task.num_variables())
            .flat_map(|var| {
                let domain = task.variables[var].domain_size;
                let relaxation = lm_cut.relaxation();
                (0..domain).map(move |value| relaxation.fact_id(Fact::new(var, value)))
            })
            .collect();
        let mut goal = vec![None; task.num_variables()];
        for fact in &task.goal {
            goal[fact.var] = Some(usize::try_from(fact.value).expect("value fits usize"));
        }

        Self(Mode::Seeded(Box::new(Seeded {
            lm_cut: RefCell::new((lm_cut, Vec::new())),
            h_max: HMax::new(task),
            offsets,
            ids,
            goal,
        })))
    }

    /// The price-level evaluation on `task` (see [`LmCutHeuristic`]): a
    /// task and an `LmCut` per estimate, charging each leaf state's price
    /// once for all its facts.
    #[must_use]
    pub fn with_price_levels(task: &Task) -> Self {
        Self(Mode::Levels(Levels::new(task)))
    }
}

impl Seeded {
    /// The estimate: the larger of seeded LM-cut and `h^max`; `None` if
    /// either declares a dead end.
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64> {
        let h_max = self.h_max.estimate(facts)?;
        let representatives = self.representatives(facts);

        let mut guard = self.lm_cut.borrow_mut();
        let (lm_cut, seeds) = &mut *guard;
        seeds.clear();
        seeds.resize(lm_cut.relaxation().num_facts(), None);
        for var in 0..self.offsets.len() - 1 {
            let full = representatives.contains(&(facts.leaf_of(var), var));
            for f in self.offsets[var]..self.offsets[var + 1] {
                let price = facts.prices[f];
                if price != u64::MAX {
                    seeds[self.ids[f]] = Some(if full { price } else { 0 });
                }
            }
        }

        let lm = lm_cut.value_seeded(seeds, None)?;
        Some(lm.max(h_max))
    }

    /// Per factor (`None` is the center), its representative variable:
    /// the most expensive goal fact first, then the most expensive priced
    /// fact, then the smaller variable.
    fn representatives(&self, facts: &FactPrices<'_>) -> Vec<(Option<usize>, VarId)> {
        let mut best: Vec<(Option<usize>, RepresentativeKey)> = Vec::new();
        for var in 0..self.offsets.len() - 1 {
            let priced = |f: usize| Some(facts.prices[f]).filter(|&p| p != u64::MAX);
            let range = self.offsets[var]..self.offsets[var + 1];
            let most = range.clone().filter_map(priced).max().unwrap_or(0);
            let goal = self.goal[var]
                .and_then(|value| priced(self.offsets[var] + value))
                .unwrap_or(0);
            let key = (goal, most, Reverse(var));
            let factor = facts.leaf_of(var);
            match best.iter_mut().find(|(f, _)| *f == factor) {
                Some((_, current)) if *current < key => *current = key,
                Some(_) => {}
                None => best.push((factor, key)),
            }
        }

        best.into_iter()
            .map(|(factor, (_, _, Reverse(var)))| (factor, var))
            .collect()
    }
}

impl DecoupledHeuristic for LmCutHeuristic {
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64> {
        match &self.0 {
            Mode::Seeded(seeded) => seeded.estimate(facts),
            Mode::Levels(levels) => levels.estimate(facts),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use sas_plus::{Effect, Operator, State, Variable};

    use super::*;
    use crate::Factoring;

    /// One leaf of two variables, both goals, reached together in one leaf
    /// state at price 5: the cheapest completion costs 5. Seeding both
    /// facts at their full price would charge 10; the shares charge 5.
    #[test]
    fn a_leaf_state_is_paid_once() {
        let set = |var| Operator {
            name: "set".into(),
            pre: vec![],
            eff: vec![Effect::new(Fact::new(var, 1))],
            cost: 5,
        };
        let task = Task {
            variables: vec![Variable::new("x", 2), Variable::new("y", 2)],
            operators: vec![set(0), set(1)],
            initial: State(vec![0, 0]),
            goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
            axioms: vec![],
        };
        let leaf = Factoring::new(&task, vec![vec![0, 1]]).expect("valid");
        let prices = FactPrices::new(&task, &leaf, |f| Some(if f.value == 1 { 5 } else { 0 }))
            .expect("valid");

        for h in [
            LmCutHeuristic::new(&task),
            LmCutHeuristic::with_price_levels(&task),
        ] {
            assert_eq!(h.estimate(&prices), Some(5));
        }
    }
}
