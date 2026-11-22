//! Generalized factorings (Gnad, Torralba and Fišer, ICAPS 2022): leaves
//! that may interact directly and a center that may be empty. Any partition
//! is searchable here (operators that are not leaf-only actions are global
//! actions), so the strategy only has to choose leaves that are mobile.

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{Task, VarId};

use super::{Factoring, leaf_states};

impl Factoring {
    /// A generalized factoring: leaves chosen among the effect variable sets
    /// of the operators (the paper's potential leaves), as many mutually
    /// disjoint mobile ones as a greedy pass finds; every other variable is
    /// the center, which may end up empty.
    ///
    /// Gnad, Torralba and Fišer (ICAPS 2022) select the leaves with an
    /// integer program maximizing the number of mobile leaves; this is a
    /// greedy approximation of that objective. Candidates go smallest first,
    /// then by how few other candidates they overlap, then by how many
    /// operators set exactly that variable set. A leaf is
    /// *mobile* if some operator sets only its variables and reads only it
    /// and the center. A candidate is kept if it is disjoint from the leaves
    /// so far and it and every earlier leaf stay mobile (a new leaf takes its
    /// variables out of the center, which can strand another leaf), so every
    /// leaf of the result is mobile. Candidates with a derived variable or
    /// more than [`crate::MAX_LEAF_STATES`] states are skipped.
    ///
    /// Leaves may read and write each other: those operators become global
    /// actions, which the search branches over (their Theorem 1 shows that
    /// this is decoupled search on a star factoring of a slightly extended
    /// task, so it stays sound, complete and optimal).
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Factoring, search};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // Two counters that each step on their own, and a swap that touches
    /// // both: no star factoring has two leaves, a generalized one has.
    /// let op = |name: &str, pre: Vec<Fact>, eff: Vec<Fact>| Operator {
    ///     name: name.into(),
    ///     pre,
    ///     eff: eff.into_iter().map(Effect::new).collect(),
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![
    ///         op("a", vec![Fact::new(0, 0)], vec![Fact::new(0, 1)]),
    ///         op("b", vec![Fact::new(1, 0)], vec![Fact::new(1, 1)]),
    ///         op("swap", vec![Fact::new(0, 1)], vec![Fact::new(0, 0), Fact::new(1, 0)]),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(0, 1), Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let factoring = Factoring::generalized(&task);
    /// assert!(factoring.center().is_empty());
    /// assert_eq!(factoring.leaves(), &[vec![0], vec![1]]);
    /// assert_eq!(search(&task, &factoring)?.map(|p| p.cost), Some(2));
    /// # Ok::<(), decoupled_search::Error>(())
    /// ```
    #[must_use]
    pub fn generalized(task: &Task) -> Self {
        let mut leaves: Vec<Vec<VarId>> = Vec::new();
        for candidate in candidates(task) {
            let free = candidate
                .iter()
                .all(|v| leaves.iter().all(|leaf| !leaf.contains(v)));
            if !free {
                continue;
            }

            leaves.push(candidate);
            if !(0..leaves.len()).all(|l| is_mobile(task, &leaves, l)) {
                leaves.pop();
            }
        }

        leaves.sort_unstable();
        Self::from_leaves(task, leaves)
    }
}

/// The potential leaves: every operator's effect variables, without
/// derived variables and leaves above `MAX_LEAF_STATES`. Ordered smallest
/// first, then by how few other candidates they overlap (the usual greedy
/// rule for packing many disjoint sets), then by how many operators have
/// exactly that effect set.
fn candidates(task: &Task) -> Vec<Vec<VarId>> {
    let mut counted: Vec<(Vec<VarId>, usize)> = Vec::new();
    for op in &task.operators {
        let mut vars: Vec<VarId> = op.eff.iter().map(|e| e.fact.var).collect();
        vars.sort_unstable();
        vars.dedup();
        if vars.is_empty()
            || vars.iter().any(|&v| task.variables[v].is_derived())
            || leaf_states(task, &vars).is_none()
        {
            continue;
        }

        match counted.iter_mut().find(|(c, _)| *c == vars) {
            Some((_, n)) => *n += 1,
            None => counted.push((vars, 1)),
        }
    }

    let overlaps = |a: &[VarId]| {
        counted
            .iter()
            .filter(|(b, _)| b.as_slice() != a && b.iter().any(|v| a.contains(v)))
            .count()
    };
    let mut keyed: Vec<_> = counted
        .iter()
        .map(|(vars, n)| {
            (
                vars.len(),
                overlaps(vars),
                core::cmp::Reverse(*n),
                vars.clone(),
            )
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(.., vars)| vars).collect()
}

/// Whether leaf `index` of `leaves` is mobile: some operator sets only its
/// variables and reads only it and the center (the variables in no leaf).
fn is_mobile(task: &Task, leaves: &[Vec<VarId>], index: usize) -> bool {
    let mut owner = vec![None; task.num_variables()];
    for (l, leaf) in leaves.iter().enumerate() {
        for &v in leaf {
            owner[v] = Some(l);
        }
    }

    task.operators.iter().any(|op| {
        !op.eff.is_empty()
            && op.eff.iter().all(|e| owner[e.fact.var] == Some(index))
            && op
                .pre
                .iter()
                .chain(op.eff.iter().flat_map(|e| &e.conditions))
                .all(|f| owner[f.var].is_none_or(|l| l == index))
    })
}
