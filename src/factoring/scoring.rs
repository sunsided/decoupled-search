//! Scoring factorings, and choosing among the strategies automatically.
//!
//! A leaf is *mobile* if it has at least one leaf action. This is Gnad,
//! Poser and Hoffmann's (IJCAI 2017) definition: their flexibility of a
//! leaf is the share of the actions affecting it that affect only it, and
//! a leaf is mobile if that share is positive; for strict-star factorings
//! the actions affecting only a leaf are exactly our leaf actions. Like
//! their strategies (and Torralba, Gnad, Dubbert and Hoffmann 2016),
//! [`Factoring::auto`] abstains, falling back to explicit search, when no
//! candidate has two mobile leaves: the gain of decoupled search comes from
//! not enumerating combinations of leaf states across leaves.
//!
//! The paper ranks the candidate factorings by one-second sample searches
//! (expansions for optimal planning, heuristic improvement for satisficing
//! planning). This crate has no clock (`no_std`) and `auto` does not know
//! the search configuration, so it ranks by the static score instead: the
//! leaf-action share, then mobile leaves, then the smaller largest leaf.
//! That ranking is ours.

use core::cmp::Reverse;

use sas_plus::Task;

use super::Factoring;
use crate::Error;
use crate::decoupled::DecoupledTask;

/// A factoring strategy of [`Factoring`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strategy {
    /// [`Factoring::fork`].
    Fork,
    /// [`Factoring::inverted_fork`].
    InvertedFork,
    /// [`Factoring::incident_arcs`].
    IncidentArcs,
    /// [`Factoring::max_leaves`], the independent-set strategy.
    MaxLeaves,
}

impl Strategy {
    /// Every strategy, in the order [`Factoring::auto`] breaks ties by.
    pub const ALL: [Self; 4] = [
        Self::Fork,
        Self::InvertedFork,
        Self::IncidentArcs,
        Self::MaxLeaves,
    ];
}

/// How well a factoring decouples a task.
///
/// # Examples
///
/// ```
/// use decoupled_search::Factoring;
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// // A truck (variable 0) and two packages that ride along.
/// let op = |pre: Vec<Fact>, eff: Fact| Operator {
///     name: "op".into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 1,
/// };
/// let task = Task {
///     variables: vec![Variable::new("truck", 2), Variable::new("p", 3), Variable::new("q", 3)],
///     operators: vec![
///         op(vec![Fact::new(0, 0)], Fact::new(0, 1)),
///         op(vec![Fact::new(0, 0), Fact::new(1, 0)], Fact::new(1, 2)),
///         op(vec![Fact::new(0, 1), Fact::new(1, 2)], Fact::new(1, 1)),
///         op(vec![Fact::new(0, 0), Fact::new(2, 0)], Fact::new(2, 2)),
///     ],
///     initial: State(vec![0, 0, 0]),
///     goal: vec![Fact::new(1, 1), Fact::new(2, 2)],
///     axioms: vec![],
/// };
///
/// let score = Factoring::fork(&task).score(&task)?;
/// assert_eq!((score.leaves, score.mobile_leaves), (2, 2));
/// assert_eq!((score.leaf_actions, score.global_actions), (3, 1));
/// assert_eq!(score.leaf_action_share(), 0.75);
/// assert_eq!((score.max_leaf_states, score.total_leaf_states), (3, 6));
/// # Ok::<(), decoupled_search::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FactoringScore {
    /// Number of leaves.
    pub leaves: usize,
    /// Number of mobile leaves: leaves with at least one leaf action, which
    /// can move without a global action (Gnad, Poser and Hoffmann 2017:
    /// flexibility above 0).
    pub mobile_leaves: usize,
    /// Operators priced inside a leaf.
    pub leaf_actions: usize,
    /// Operators searched explicitly.
    pub global_actions: usize,
    /// States of the largest leaf; 0 without leaves.
    pub max_leaf_states: usize,
    /// States of all leaves together.
    pub total_leaf_states: usize,
}

impl FactoringScore {
    /// The share of operators that are leaf actions, in `0.0..=1.0`; 0 for
    /// a task without operators.
    #[must_use]
    pub fn leaf_action_share(&self) -> f64 {
        let total = self.leaf_actions + self.global_actions;
        if total == 0 {
            return 0.0;
        }

        // Operator counts stay far below 2^52, so the conversion is exact.
        #[expect(clippy::cast_precision_loss, reason = "operator counts are small")]
        let share = self.leaf_actions as f64 / total as f64;
        share
    }

    /// The ranking key of [`Factoring::auto`], greater is better: the
    /// leaf-action share (compared exactly, as a fraction), then more
    /// mobile leaves, then the smaller largest leaf.
    fn key(&self) -> (Share, usize, Reverse<usize>) {
        (
            Share(self.leaf_actions, self.leaf_actions + self.global_actions),
            self.mobile_leaves,
            Reverse(self.max_leaf_states),
        )
    }
}

/// A fraction `numerator / denominator`, ordered by value; `0 / 0` counts
/// as 0.
#[derive(Clone, Copy, Debug)]
struct Share(usize, usize);

impl Share {
    /// The value as a pair that cross-multiplies exactly.
    fn parts(self) -> (u128, u128) {
        let widen = |n: usize| u128::try_from(n).expect("usize fits in u128");
        (widen(self.0), widen(self.1.max(1)))
    }
}

impl PartialEq for Share {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Share {}

impl PartialOrd for Share {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Share {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let ((a, b), (c, d)) = (self.parts(), other.parts());
        (a * d).cmp(&(c * b))
    }
}

impl Factoring {
    /// The factoring that `strategy` computes for `task`.
    #[must_use]
    pub fn with_strategy(task: &Task, strategy: Strategy) -> Self {
        match strategy {
            Strategy::Fork => Self::fork(task),
            Strategy::InvertedFork => Self::inverted_fork(task),
            Strategy::IncidentArcs => Self::incident_arcs(task),
            Strategy::MaxLeaves => Self::max_leaves(task),
        }
    }

    /// Scores this factoring on `task`, classifying operators exactly as
    /// the search does.
    ///
    /// # Errors
    ///
    /// Every error of [`crate::search()`] except
    /// [`Error::ReachabilityNotOptimal`]: the factoring must be searchable.
    pub fn score(&self, task: &Task) -> Result<FactoringScore, Error> {
        let decoupled = DecoupledTask::new(task, self)?;
        let leaves = &decoupled.leaves;

        Ok(FactoringScore {
            leaves: leaves.len(),
            mobile_leaves: leaves.iter().filter(|l| !l.actions.is_empty()).count(),
            leaf_actions: leaves.iter().map(|l| l.actions.len()).sum(),
            global_actions: decoupled.globals.len(),
            max_leaf_states: leaves.iter().map(|l| l.size).max().unwrap_or(0),
            total_leaf_states: leaves.iter().map(|l| l.size).sum(),
        })
    }

    /// The best factoring of every [`Strategy`] for `task` with at least two
    /// mobile leaves, by score: the largest share of leaf actions, then more
    /// mobile leaves, then the smaller largest leaf, then the earlier
    /// strategy in [`Strategy::ALL`].
    ///
    /// Factorings that [`Factoring::score`] rejects (for example because
    /// an axiom spans two of their leaves) are skipped. If no candidate has
    /// two mobile leaves, the result is the factoring without leaves:
    /// explicit search over the whole task. Gnad, Poser and Hoffmann (2017)
    /// abstain in that case too, since decoupling gains by not combining
    /// the states of several leaves; they rank candidates by short sample
    /// searches, this ranking is static (see the module docs).
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Factoring, search};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A hub that both spokes read and that reads both spokes.
    /// let op = |pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("hub", 2), Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![
    ///         op(vec![Fact::new(1, 1), Fact::new(2, 1)], Fact::new(0, 1)),
    ///         op(vec![Fact::new(0, 0)], Fact::new(1, 1)),
    ///         op(vec![Fact::new(0, 0)], Fact::new(2, 1)),
    ///     ],
    ///     initial: State(vec![0, 0, 0]),
    ///     goal: vec![Fact::new(0, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// // The fork factoring finds no sink component; the star strategies do.
    /// let factoring = Factoring::auto(&task);
    /// assert_eq!(factoring.leaves(), &[vec![1], vec![2]]);
    /// assert_eq!(search(&task, &factoring)?.map(|p| p.cost), Some(3));
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    #[must_use]
    pub fn auto(task: &Task) -> Self {
        Strategy::ALL
            .iter()
            .rev()
            .filter_map(|&s| {
                let factoring = Self::with_strategy(task, s);
                factoring.score(task).ok().map(|score| (score, factoring))
            })
            .filter(|(score, _)| score.mobile_leaves >= 2)
            .max_by_key(|(score, _)| score.key())
            .map_or_else(
                || Self::from_leaves(task, alloc::vec::Vec::new()),
                |(_, f)| f,
            )
    }
}
