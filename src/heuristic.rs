//! Heuristics on decoupled states.
//!
//! A decoupled state stands for every explicit state that combines its
//! center with one reached leaf state per leaf, each leaf state bought at its
//! price. A heuristic sees this through [`FactPrices`]: the cheapest price at
//! which each fact holds in some member state (0 for center facts). Seeding a
//! delete relaxation with those prices instead of a single state gives
//! [`HMax`] (admissible) and [`HFf`] (inadmissible, for greedy search),
//! implemented here without a dependency on a heuristics crate.
//!
//! ```mermaid
//! graph LR
//!     state[decoupled state] --> facts[FactPrices: cheapest price per fact]
//!     facts --> relaxed[relaxed exploration over all operators]
//!     relaxed --> hmax[HMax: max over goal costs]
//!     relaxed --> hff[HFf: relaxed plan cost]
//! ```

#[cfg(feature = "merge-and-shrink")]
mod mas;
mod relaxed;

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{Fact, State, Task, VarId};

use crate::{Error, Factoring};
#[cfg(feature = "merge-and-shrink")]
pub use mas::MasHeuristic;
use relaxed::Relaxation;

/// The cheapest price at which each fact holds in some member state of a
/// decoupled state.
///
/// Center facts that hold have price 0. A leaf fact has the cheapest price
/// of a reached leaf state containing it. Facts no member state contains
/// have no price.
///
/// Each variable also records its factor ([`FactPrices::leaf_of`]). A
/// member state costs at least the sum, over the factors (the center counts
/// as one), of the largest price of its facts in that factor: two facts of
/// one leaf are bought together with one leaf state, facts of different
/// leaves with different ones. Heuristics may rely on that bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactPrices<'a> {
    /// Index of each variable's first value in [`FactPrices::prices`], plus
    /// a final entry with the number of facts.
    pub(crate) offsets: Cow<'a, [usize]>,
    /// The leaf of each variable, `None` for center variables.
    pub(crate) leaves: Cow<'a, [Option<usize>]>,
    /// The price of every fact, by `offsets[var] + value`; `u64::MAX` for
    /// facts without a price.
    pub(crate) prices: Vec<u64>,
}

impl FactPrices<'_> {
    /// The same facts with every price set to 0: what a root heuristic of
    /// anytime A* sees, since it estimates only the remaining global cost.
    pub(crate) fn reached_for_free(mut self) -> Self {
        for price in &mut self.prices {
            if *price != u64::MAX {
                *price = 0;
            }
        }

        self
    }
}

impl FactPrices<'static> {
    /// Fact prices for `task` under `factoring`, with `price(fact)` as the
    /// price of every fact (`None` for facts no member state contains).
    ///
    /// Meant for testing heuristics without running a search. The caller
    /// is responsible for the prices being consistent with some decoupled
    /// state, in the sense of the bound described on [`FactPrices`].
    /// `Some(u64::MAX)` counts as no price.
    ///
    /// # Errors
    ///
    /// [`Error::FactoringMismatch`] if `factoring` does not partition exactly
    /// the variables of `task`.
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{DecoupledHeuristic, FactPrices, Factoring, HMax};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// let task = Task {
    ///     variables: vec![Variable::new("x", 2)],
    ///     operators: vec![Operator {
    ///         name: "set".into(),
    ///         pre: vec![Fact::new(0, 0)],
    ///         eff: vec![Effect::new(Fact::new(0, 1))],
    ///         cost: 3,
    ///     }],
    ///     initial: State(vec![0]),
    ///     goal: vec![Fact::new(0, 1)],
    ///     axioms: vec![],
    /// };
    /// let leaf = Factoring::new(&task, vec![vec![0]])?;
    ///
    /// // x = 0 costs 2 to reach in some member state, x = 1 is not reached.
    /// let prices = FactPrices::new(&task, &leaf, |f| (f.value == 0).then_some(2))?;
    /// assert_eq!(prices.get(Fact::new(0, 0)), Some(2));
    /// assert_eq!(prices.leaf_of(0), Some(0));
    /// assert_eq!(HMax::new(&task).estimate(&prices), Some(5));
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    pub fn new(
        task: &Task,
        factoring: &Factoring,
        mut price: impl FnMut(Fact) -> Option<u64>,
    ) -> Result<Self, Error> {
        let leaves = factoring.leaf_of(task)?;
        let offsets = fact_offsets(task);
        let prices = task
            .variables
            .iter()
            .enumerate()
            .flat_map(|(var, v)| (0..v.domain_size).map(move |value| Fact::new(var, value)))
            .map(|fact| price(fact).unwrap_or(u64::MAX))
            .collect();

        Ok(Self {
            offsets: Cow::Owned(offsets),
            leaves: Cow::Owned(leaves),
            prices,
        })
    }

    /// The fact prices of a single explicit state: its facts at price 0, all
    /// variables in the center.
    ///
    /// # Panics
    ///
    /// Panics if `state` does not have one value per variable of `task`.
    #[must_use]
    pub fn from_state(task: &Task, state: &State) -> Self {
        assert_eq!(
            state.0.len(),
            task.num_variables(),
            "the state must assign every variable of the task"
        );
        Self {
            offsets: Cow::Owned(fact_offsets(task)),
            leaves: Cow::Owned(vec![None; task.num_variables()]),
            prices: task
                .variables
                .iter()
                .zip(&state.0)
                .flat_map(|(v, &value)| {
                    (0..v.domain_size).map(move |d| if d == value { 0 } else { u64::MAX })
                })
                .collect(),
        }
    }
}

impl FactPrices<'_> {
    /// The cheapest price at which `fact` holds in a member state; `None` if
    /// none contains it or the fact is out of range.
    #[must_use]
    pub fn get(&self, fact: Fact) -> Option<u64> {
        let value = usize::try_from(fact.value).ok()?;
        let index = self.offsets.get(fact.var)? + value;
        let next = self.offsets.get(fact.var + 1)?;
        (index < *next)
            .then(|| self.prices[index])
            .filter(|&p| p != u64::MAX)
    }

    /// The leaf that variable `var` belongs to, as an index into
    /// [`Factoring::leaves`]; `None` for center variables and for variables
    /// out of range.
    #[must_use]
    pub fn leaf_of(&self, var: VarId) -> Option<usize> {
        self.leaves.get(var).copied().flatten()
    }

    /// The number of variables these prices cover.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.leaves.len()
    }
}

/// The index of every variable's first value in a vector over all facts of
/// `task`, plus a final entry with the number of facts.
pub(crate) fn fact_offsets(task: &Task) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(task.num_variables() + 1);
    let mut next = 0;
    offsets.push(next);
    for v in &task.variables {
        next += usize::try_from(v.domain_size).expect("a u32 fits in usize");
        offsets.push(next);
    }

    offsets
}

/// A heuristic on decoupled states.
pub trait DecoupledHeuristic {
    /// An estimate of the cost still to pay from a decoupled state beyond
    /// the cost of its global actions: leaf prices plus all further
    /// operators. `None` declares that no goal is reachable from it; the
    /// search then prunes the state.
    ///
    /// For [`crate::astar`] to stay optimal, the estimate must not exceed
    /// the true remaining cost of the cheapest completion through any member
    /// state, and `None` must only be returned when none exists.
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64>;
}

impl<H: DecoupledHeuristic + ?Sized> DecoupledHeuristic for &H {
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64> {
        (**self).estimate(facts)
    }
}

/// The max heuristic `h^max` (Bonet and Geffner 2001) seeded with fact
/// prices: admissible on decoupled states.
///
/// A completion through a member state pays the prices of its leaf states,
/// at least the largest price of any fact it uses, and then at least `h^max`
/// of that state; seeding every fact with its price can only lower the
/// estimate. The relaxation runs over all operators of the original task,
/// leaf actions included.
///
/// # Examples
///
/// ```
/// use decoupled_search::{Factoring, HMax, astar};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// let op = |pre: Vec<Fact>, eff: Fact| Operator {
///     name: "op".into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 2,
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
/// let plan = astar(&task, &Factoring::fork(&task), &HMax::new(&task))?.expect("solvable");
/// assert_eq!(plan.cost, 6);
/// # Ok::<(), decoupled_search::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct HMax(Relaxation);

impl HMax {
    /// Prepares `h^max` for `task`.
    ///
    /// # Panics
    ///
    /// Panics if `task` fails [`Task::validate`].
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self(Relaxation::new(task))
    }
}

impl DecoupledHeuristic for HMax {
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64> {
        self.0.h_max(facts)
    }
}

/// The FF heuristic `h^FF` (Hoffmann and Nebel 2001) seeded with fact
/// prices: the cost of a relaxed plan built from `h^add` best supporters,
/// plus the largest price of a seed fact it uses. Not admissible; meant for
/// [`crate::gbfs`].
///
/// # Examples
///
/// ```
/// use decoupled_search::{Factoring, HFf, gbfs};
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
/// let plan = gbfs(&task, &Factoring::fork(&task), &HFf::new(&task))?.expect("solvable");
/// assert_eq!(task.validate_plan(&plan.ops), Some(plan.cost));
/// # Ok::<(), decoupled_search::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct HFf(Relaxation);

impl HFf {
    /// Prepares `h^FF` for `task`.
    ///
    /// # Panics
    ///
    /// Panics if `task` fails [`Task::validate`].
    #[must_use]
    pub fn new(task: &Task) -> Self {
        Self(Relaxation::new(task))
    }
}

impl DecoupledHeuristic for HFf {
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64> {
        self.0.h_ff(facts)
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::VecDeque;
    use alloc::vec;
    use alloc::vec::Vec;

    use sas_plus::random::{RandomTaskParams, random_task};

    use super::*;
    use crate::decoupled::DecoupledTask;

    /// Checks on up to 40 decoupled states of `task` under `factoring`
    /// (breadth-first from the initial state) that `heuristic` never
    /// exceeds the cheapest completion through a member state, and declares
    /// dead ends only where no member state can reach the goal. Returns the
    /// number of states with a finite estimate above 0.
    fn check_admissible(
        task: &Task,
        factoring: &Factoring,
        heuristic: &dyn DecoupledHeuristic,
    ) -> u64 {
        let decoupled = DecoupledTask::new(task, factoring).expect("supported");
        let perfect = task.perfect_heuristic();
        let mut informative = 0;
        let mut queue = VecDeque::from([decoupled.initial()]);
        let mut seen = Vec::new();
        while let Some(state) = queue.pop_front() {
            if seen.len() >= 40 || seen.contains(&state) {
                continue;
            }

            let truth = decoupled
                .member_states(task, &state)
                .iter()
                .filter_map(|(member, price)| perfect(member).map(|h| h + price))
                .min();
            let estimate = heuristic.estimate(&decoupled.fact_prices(&state));
            match (estimate, truth) {
                (None, None) => {}
                (None, Some(t)) => panic!("dead end declared, but {t} is reachable"),
                (Some(h), None) => assert!(h < u64::MAX),
                (Some(h), Some(t)) => assert!(h <= t, "{h} > {t} on {state:?}"),
            }

            informative += u64::from(estimate.is_some_and(|h| h > 0));
            queue.extend(decoupled.successors(&state).into_iter().map(|(_, s)| s));
            seen.push(state);
        }

        informative
    }

    /// The factorings to check on `task`: fork, single-variable leaves and
    /// pairs of state variables.
    fn factorings(task: &Task) -> Vec<Factoring> {
        let vars: Vec<usize> = (0..task.num_variables())
            .filter(|&v| !task.variables[v].is_derived())
            .collect();
        let singles = vars.iter().map(|&v| vec![v]).collect();
        let pairs = vars.chunks(2).map(<[usize]>::to_vec).collect();
        [singles, pairs]
            .into_iter()
            .filter_map(|leaves| Factoring::new(task, leaves).ok())
            .chain([Factoring::fork(task)])
            .collect()
    }

    /// Random tasks, with derived variables in half of them.
    fn tasks() -> impl Iterator<Item = Task> {
        (0..60).map(|seed| {
            let params = RandomTaskParams {
                num_derived: usize::from(seed % 2 == 0),
                ..RandomTaskParams::default()
            };
            random_task(seed, params)
        })
    }

    /// `h^max` on fact prices is admissible on decoupled states.
    #[test]
    fn h_max_is_admissible_on_member_states() {
        let mut informative = 0;
        for task in tasks() {
            let h = HMax::new(&task);
            for factoring in factorings(&task) {
                if DecoupledTask::new(&task, &factoring).is_ok() {
                    informative += check_admissible(&task, &factoring, &h);
                }
            }
        }

        assert!(
            informative > 100,
            "only {informative} informative estimates"
        );
    }

    /// The merge-and-shrink adapter is admissible on decoupled states, with
    /// exact bisimulation, a lossy size limit, and no shrinking at all.
    #[cfg(feature = "merge-and-shrink")]
    #[test]
    fn merge_and_shrink_is_admissible_on_member_states() {
        use merge_and_shrink::{Bisimulation, LinearMerge, MergeAndShrink, NoShrink};

        let mut informative = 0;
        for task in tasks() {
            let linear = || LinearMerge::in_variable_order(task.num_variables());
            let heuristics = [
                MasHeuristic::build(&task, MergeAndShrink::new(linear(), Bisimulation)),
                MasHeuristic::build(
                    &task,
                    MergeAndShrink {
                        max_states: 4,
                        ..MergeAndShrink::new(linear(), Bisimulation)
                    },
                ),
                MasHeuristic::build(
                    &task,
                    MergeAndShrink {
                        prune: false,
                        ..MergeAndShrink::new(linear(), NoShrink)
                    },
                ),
            ];
            for factoring in factorings(&task) {
                if DecoupledTask::new(&task, &factoring).is_err() {
                    continue;
                }

                for h in &heuristics {
                    informative += check_admissible(&task, &factoring, h);
                }
            }
        }

        assert!(
            informative > 100,
            "only {informative} informative estimates"
        );
    }
}
