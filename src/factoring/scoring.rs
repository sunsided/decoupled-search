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

use alloc::vec::Vec;
use core::cmp::Reverse;

use sas_plus::Task;

use super::Factoring;
use crate::Error;
use crate::decoupled::DecoupledTask;
use crate::search;

/// A factoring strategy of [`Factoring`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strategy {
    /// [`Factoring::fork`].
    Fork,
    /// [`Factoring::inverted_fork`].
    InvertedFork,
    /// [`Factoring::x_shape`].
    XShape,
    /// [`Factoring::incident_arcs`].
    IncidentArcs,
    /// [`Factoring::max_leaves`], the independent-set strategy.
    MaxLeaves,
}

impl Strategy {
    /// Every strategy, in the order [`Factoring::auto`] breaks ties by.
    pub const ALL: [Self; 5] = [
        Self::Fork,
        Self::InvertedFork,
        Self::XShape,
        Self::IncidentArcs,
        Self::MaxLeaves,
    ];
}

/// What [`Factoring::auto_by_sample`] selects for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Objective {
    /// Cost-optimal planning: samples run decoupled A* with `h^max`.
    Optimal,
    /// Satisficing planning: samples run greedy best-first search with
    /// `h^FF`.
    Satisficing,
}

/// The measurements of one sample search, compared by
/// [`Factoring::auto_by_sample`].
#[derive(Clone, Copy, Debug)]
struct Sample {
    /// Whether the sample finished within its budget: proved its result
    /// (cost-optimal), or found a plan or proved there is none
    /// (satisficing).
    solved: bool,
    /// Expansions made.
    expanded: u64,
    /// Work units spent, at least 1.
    work: u64,
    /// Heuristic value of the initial state.
    initial_h: u64,
    /// Least heuristic value seen.
    best_h: u64,
}

impl Sample {
    /// Runs the sample search of `objective` on `factoring` for at most
    /// `expansions` expansions; `None` if the search rejects the factoring.
    fn run(
        task: &Task,
        factoring: &Factoring,
        objective: Objective,
        expansions: u64,
    ) -> Option<Self> {
        let optimal = objective == Objective::Optimal;
        let (probe, expanded) = search::sample(task, factoring, optimal, expansions).ok()?;
        let solved = !probe.cut_off;
        Some(Self {
            solved,
            expanded,
            work: probe.work.max(1),
            initial_h: probe.initial_h.unwrap_or(0),
            best_h: probe.best_h.unwrap_or(0),
        })
    }

    /// Whether this sample ranks strictly above `other` under `objective`:
    /// finished samples first, the less work among them; then, for
    /// cost-optimal planning, more expansions per work unit (the paper's
    /// expansions within a time limit), and for satisficing planning, the
    /// larger heuristic improvement `initial_h / best_h`, then less work.
    fn beats(&self, other: &Self, objective: Objective) -> bool {
        let wide = u128::from;
        if self.solved != other.solved {
            return self.solved;
        }

        if self.solved {
            return self.work < other.work;
        }

        match objective {
            Objective::Optimal => {
                wide(self.expanded) * wide(other.work) > wide(other.expanded) * wide(self.work)
            }
            Objective::Satisficing => {
                let ours = wide(self.initial_h) * wide(other.best_h.max(1));
                let theirs = wide(other.initial_h) * wide(self.best_h.max(1));
                ours > theirs || (ours == theirs && self.work < other.work)
            }
        }
    }
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
            Strategy::XShape => Self::x_shape(task),
            Strategy::IncidentArcs => Self::incident_arcs(task),
            Strategy::MaxLeaves => Self::max_leaves(task),
        }
    }

    /// `task` with every operator that is a leaf action under this
    /// factoring at cost 0: the task whose heuristics are root heuristics
    /// for [`crate::Config::anytime_astar`] (Gnad and Hoffmann's
    /// leaves-for-free root estimate, ICAPS 2015, Definition 6).
    ///
    /// # Errors
    ///
    /// As [`Factoring::score`].
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::Factoring;
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// let op = |pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 2,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("truck", 2), Variable::new("pkg", 2)],
    ///     operators: vec![
    ///         op(vec![Fact::new(0, 0)], Fact::new(0, 1)),
    ///         op(vec![Fact::new(0, 1)], Fact::new(1, 1)),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// // The package moves by a leaf action, the truck by a global one.
    /// let free = Factoring::fork(&task).with_free_leaf_actions(&task)?;
    /// let costs: Vec<u32> = free.operators.iter().map(|o| o.cost).collect();
    /// assert_eq!(costs, [2, 0]);
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    pub fn with_free_leaf_actions(&self, task: &Task) -> Result<Task, Error> {
        let decoupled = DecoupledTask::new(task, self)?;
        let mut free = task.clone();
        for action in decoupled.leaves.iter().flat_map(|l| &l.actions) {
            free.operators[action.op].cost = 0;
        }

        Ok(free)
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

    /// The factoring of the strategy whose short sample search does best,
    /// among those [`Factoring::auto`] would consider (two mobile leaves,
    /// accepted by the search): Gnad, Poser and Hoffmann's (2017)
    /// per-instance selection, with an expansion budget instead of a time
    /// limit.
    ///
    /// Each distinct candidate runs a sample search of `objective` (see
    /// [`Objective`]) for at most `expansions` expansions. A sample that
    /// finishes (cost-optimal: proves its result; satisficing: finds a
    /// plan) wins, the cheapest such one first; the paper would return its
    /// plan, here the caller searches again. Otherwise, as in the paper,
    /// satisficing planning prefers the larger heuristic improvement (the
    /// initial state's `h^FF` over the least one seen), and cost-optimal
    /// planning the faster expansion rate. With a budget of expansions
    /// rather than of time, the rate is measured in deterministic work
    /// units (per generated decoupled state, one plus the leaf states its
    /// leaf closures range over), our proxy for time. Ties go to the earlier
    /// strategy in [`Strategy::ALL`]. Without a candidate, the result is
    /// the factoring without leaves; with one, no sample is run.
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Factoring, Objective, search};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A truck (0) and two packages that ride along.
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
    ///         op(vec![Fact::new(0, 1), Fact::new(2, 2)], Fact::new(2, 1)),
    ///     ],
    ///     initial: State(vec![0, 0, 0]),
    ///     goal: vec![Fact::new(1, 1), Fact::new(2, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let factoring = Factoring::auto_by_sample(&task, Objective::Optimal, 100);
    /// assert_eq!(factoring.leaves().len(), 2);
    /// assert_eq!(search(&task, &factoring)?.map(|p| p.cost), Some(5));
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    #[must_use]
    pub fn auto_by_sample(task: &Task, objective: Objective, expansions: u64) -> Self {
        let mut candidates: Vec<Self> = Vec::new();
        for &strategy in &Strategy::ALL {
            let factoring = Self::with_strategy(task, strategy);
            let usable = factoring
                .score(task)
                .is_ok_and(|score| score.mobile_leaves >= 2);
            if usable && !candidates.contains(&factoring) {
                candidates.push(factoring);
            }
        }

        if candidates.len() <= 1 {
            return candidates
                .pop()
                .unwrap_or_else(|| Self::from_leaves(task, Vec::new()));
        }

        let mut best: Option<(Sample, Self)> = None;
        for factoring in candidates {
            let Some(sample) = Sample::run(task, &factoring, objective, expansions) else {
                continue;
            };

            if best
                .as_ref()
                .is_none_or(|(b, _)| sample.beats(b, objective))
            {
                best = Some((sample, factoring));
            }
        }

        best.map_or_else(|| Self::from_leaves(task, Vec::new()), |(_, f)| f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample with the given measurements.
    fn sample(solved: bool, expanded: u64, work: u64, (initial_h, best_h): (u64, u64)) -> Sample {
        Sample {
            solved,
            expanded,
            work,
            initial_h,
            best_h,
        }
    }

    /// Finished samples win, the cheaper first; then the expansion rate
    /// (optimal) or the heuristic improvement (satisficing).
    #[test]
    fn samples_rank_as_the_paper_does() {
        let (optimal, satisficing) = (Objective::Optimal, Objective::Satisficing);
        let done = sample(true, 5, 50, (4, 0));
        let faster = sample(false, 100, 1_000, (4, 2));
        let slower = sample(false, 100, 3_000, (8, 2));
        assert!(done.beats(&faster, optimal) && !faster.beats(&done, optimal));
        assert!(sample(true, 9, 40, (4, 0)).beats(&done, optimal));

        assert!(faster.beats(&slower, optimal) && !slower.beats(&faster, optimal));
        // 8 / 2 beats 4 / 2 although it is slower.
        assert!(slower.beats(&faster, satisficing) && !faster.beats(&slower, satisficing));
        assert!(!faster.beats(&faster, satisficing), "ties do not win");
    }
}
