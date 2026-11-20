//! Cross-checks decoupled search against explicit Dijkstra on random tasks
//! under several factorings.

use decoupled_search::{Error, Factoring, search};
use sas_plus::Task;
use sas_plus::random::{RandomTaskParams, random_task};

use crate::checked_cost;
use crate::dijkstra::optimal_cost;
use crate::star::strategies;

/// Number of random tasks per test.
const SEEDS: u64 = 200;

/// Leaf sets over the state variables `vars` (derived variables cannot be
/// leaf variables): the fork factoring is added separately. Random tasks
/// mostly have one causal-graph component, so the fork factoring alone would
/// rarely produce leaves.
fn partitions(vars: &[usize]) -> Vec<Vec<Vec<usize>>> {
    let mut all = vec![
        Vec::new(),
        vars.iter().map(|&v| vec![v]).collect(),
        vars.iter().skip(1).map(|&v| vec![v]).collect(),
        vars.chunks(2).map(<[usize]>::to_vec).collect(),
    ];
    if !vars.is_empty() {
        all.push(vec![vars.to_vec()]);
    }

    all
}

/// Every factoring tried on `task`.
pub fn factorings(task: &Task) -> Vec<Factoring> {
    let vars: Vec<usize> = (0..task.num_variables())
        .filter(|&v| !task.variables[v].is_derived())
        .collect();
    let explicit = partitions(&vars)
        .into_iter()
        .map(|leaves| Factoring::new(task, leaves).expect("valid partition"));
    std::iter::once(Factoring::fork(task))
        .chain(explicit)
        .collect()
}

/// Checks every factoring of the random task for `seed` against Dijkstra and
/// returns how many solvable, nonzero-cost runs had at least two leaves.
fn check_seed(seed: u64, params: RandomTaskParams) -> u64 {
    check_task(seed, &random_task(seed, params))
}

/// Checks every factoring of `task` (generated from `seed`) against Dijkstra
/// and returns how many solvable, nonzero-cost runs had at least two leaves.
fn check_task(seed: u64, task: &Task) -> u64 {
    let expected = optimal_cost(task);
    let mut decoupled_runs = 0;
    for factoring in factorings(task) {
        assert_eq!(
            checked_cost(task, &factoring),
            expected,
            "seed {seed}, factoring {factoring:?}"
        );
        if factoring.leaves().len() >= 2 && expected.is_some_and(|c| c > 0) {
            decoupled_runs += 1;
        }
    }

    decoupled_runs
}

/// Default-sized random tasks under every factoring.
#[test]
fn optimal_costs_match_dijkstra_on_small_tasks() {
    let params = RandomTaskParams::default();
    let runs: u64 = (0..SEEDS).map(|seed| check_seed(seed, params)).sum();
    assert!(runs > SEEDS, "only {runs} multi-leaf runs");
}

/// Six-variable random tasks under every factoring.
#[test]
fn optimal_costs_match_dijkstra_on_larger_tasks() {
    let params = RandomTaskParams {
        num_vars: 6,
        max_domain: 3,
        num_ops: 16,
        max_cost: 4,
        ..RandomTaskParams::default()
    };
    let runs: u64 = (0..SEEDS).map(|seed| check_seed(seed, params)).sum();
    assert!(runs > SEEDS, "only {runs} multi-leaf runs");
}

/// Zero-cost operators create price ties, which leaf closure and plan
/// reconstruction must resolve without cycles.
#[test]
fn optimal_costs_match_dijkstra_with_zero_costs() {
    let params = RandomTaskParams {
        num_vars: 5,
        num_ops: 14,
        ..RandomTaskParams::default()
    };
    let runs: u64 = (0..SEEDS)
        .map(|seed| {
            let mut task = random_task(seed, params);
            for op in task.operators.iter_mut().step_by(2) {
                op.cost = 0;
            }

            check_task(seed, &task)
        })
        .sum();
    assert!(runs > SEEDS / 2, "only {runs} multi-leaf runs");
}

/// Whether `error` is one of the rejections of conditional effects and
/// axioms that the decoupled semantics cannot express.
fn is_semantic_rejection(error: &Error) -> bool {
    matches!(
        error,
        Error::CenterEffectReadsLeaf { .. }
            | Error::EffectConditionAcrossLeaves { .. }
            | Error::AxiomAcrossLeaves { .. }
    )
}

/// Random tasks with conditional effects and derived variables: every
/// factoring either is rejected with a precise error or matches Dijkstra.
/// Returns the number of multi-leaf runs with a nonzero optimum and the
/// number of rejections.
fn check_or_reject(params: RandomTaskParams) -> (u64, u64) {
    let (mut runs, mut rejected) = (0, 0);
    for seed in 0..SEEDS {
        let task = random_task(seed, params);
        let expected = optimal_cost(&task);
        let strategies = strategies(&task).into_iter().map(|(_, f)| f);
        for factoring in factorings(&task).into_iter().chain(strategies) {
            match search(&task, &factoring) {
                Err(e) if is_semantic_rejection(&e) => rejected += 1,
                Err(e) => panic!("seed {seed}: unexpected {e}"),
                Ok(_) => {
                    assert_eq!(
                        checked_cost(&task, &factoring),
                        expected,
                        "seed {seed}, {params:?}, {factoring:?}"
                    );
                    if factoring.leaves().len() >= 2 && expected.is_some_and(|c| c > 0) {
                        runs += 1;
                    }
                }
            }
        }
    }

    (runs, rejected)
}

/// Unmodified random tasks with conditional effects or derived variables:
/// most multi-leaf factorings are rejected, the rest must be right.
#[test]
fn conditional_effects_and_axioms_are_rejected_or_right() {
    for params in [
        RandomTaskParams {
            max_conditions: 2,
            ..RandomTaskParams::default()
        },
        RandomTaskParams {
            num_derived: 2,
            ..RandomTaskParams::default()
        },
    ] {
        let (runs, rejected) = check_or_reject(params);
        assert!(rejected > 0, "{params:?}: nothing was rejected");
        if params.num_derived > 0 {
            assert!(runs > 10, "{params:?}: only {runs} multi-leaf runs");
        }
    }
}

/// The factor of variable `var` under `factoring`: `Some(leaf)` or `None`
/// for the center; derived variables use `derived`, the factor chosen for
/// them so far.
fn factor_of(
    factoring: &Factoring,
    derived: &[Option<usize>],
    task: &Task,
    var: usize,
) -> Option<usize> {
    if task.variables[var].is_derived() {
        return derived[var];
    }

    factoring
        .leaves()
        .iter()
        .position(|leaf| leaf.contains(&var))
}

/// `task` with every effect condition and axiom condition dropped that
/// the decoupled semantics cannot express under `factoring`: effects on
/// the center keep only center conditions, effects on a leaf only
/// conditions on the center and that leaf, and each derived variable
/// depends on at most one leaf (the first one its rules mention). Also
/// returns how many derived variables depend on a leaf.
fn supported_variant(task: &Task, factoring: &Factoring) -> (Task, usize) {
    let mut task = task.clone();
    let mut derived = vec![None; task.num_variables()];
    let mut order: Vec<usize> = (0..task.num_variables())
        .filter(|&v| task.variables[v].is_derived())
        .collect();
    order.sort_by_key(|&v| task.variables[v].axiom_layer);
    let shape = task.clone();
    for var in order {
        let mut chosen = None;
        for axiom in task.axioms.iter_mut().filter(|a| a.head.var == var) {
            axiom
                .conditions
                .retain(|c| match factor_of(factoring, &derived, &shape, c.var) {
                    None => true,
                    Some(leaf) => *chosen.get_or_insert(leaf) == leaf,
                });
        }

        derived[var] = chosen;
    }

    for op in &mut task.operators {
        for effect in &mut op.eff {
            let target = factor_of(factoring, &derived, &shape, effect.fact.var);
            effect.conditions.retain(|c| {
                factor_of(factoring, &derived, &shape, c.var)
                    .is_none_or(|leaf| Some(leaf) == target)
            });
        }
    }

    let attached = derived.iter().filter(|d| d.is_some()).count();
    (task, attached)
}

/// Random tasks with conditional effects and derived variables, with the
/// conditions the decoupled semantics cannot express dropped per
/// factoring: every factoring is accepted and matches Dijkstra.
#[test]
fn optimal_costs_match_dijkstra_with_conditional_effects_and_axioms() {
    for params in [
        RandomTaskParams {
            max_conditions: 2,
            ..RandomTaskParams::default()
        },
        RandomTaskParams {
            num_vars: 5,
            max_conditions: 2,
            num_derived: 2,
            ..RandomTaskParams::default()
        },
    ] {
        let (mut runs, mut attached_runs) = (0, 0);
        for seed in 0..SEEDS {
            let original = random_task(seed, params);
            for factoring in factorings(&original) {
                let (task, attached) = supported_variant(&original, &factoring);
                let expected = optimal_cost(&task);
                assert_eq!(
                    checked_cost(&task, &factoring),
                    expected,
                    "seed {seed}, {params:?}, {factoring:?}"
                );
                if factoring.leaves().len() >= 2 && expected.is_some_and(|c| c > 0) {
                    runs += 1;
                    attached_runs += u64::from(attached > 0);
                }
            }
        }

        assert!(runs > SEEDS / 2, "{params:?}: only {runs} multi-leaf runs");
        if params.num_derived > 0 {
            assert!(
                attached_runs > SEEDS / 10,
                "{params:?}: only {attached_runs} runs with leaf-dependent derived variables"
            );
        }
    }
}
