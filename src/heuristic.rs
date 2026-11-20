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

mod relaxed;

use alloc::vec::Vec;

use sas_plus::{Fact, Task};

use relaxed::Relaxation;

/// The cheapest price at which each fact holds in some member state of a
/// decoupled state.
///
/// Center facts that hold have price 0. A leaf fact has the cheapest price
/// of a reached leaf state containing it. Facts no member state contains
/// have no price.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactPrices<'a> {
    /// Index of each variable's first value in [`FactPrices::prices`], plus
    /// a final entry with the number of facts.
    pub(crate) offsets: &'a [usize],
    /// The price of every fact, by `offsets[var] + value`; `u64::MAX` for
    /// facts without a price.
    pub(crate) prices: Vec<u64>,
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
