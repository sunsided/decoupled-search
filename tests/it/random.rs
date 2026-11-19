//! Cross-checks decoupled search against explicit Dijkstra on random tasks
//! under several factorings.

use decoupled_search::Factoring;
use sas_plus::Task;
use sas_plus::random::{RandomTaskParams, random_task};

use crate::checked_cost;
use crate::dijkstra::optimal_cost;

/// Number of random tasks per test.
const SEEDS: u64 = 200;

/// Leaf sets to try on a task with `n` variables: the fork factoring is
/// added separately. Random tasks mostly have one causal-graph component, so
/// the fork factoring alone would rarely produce leaves.
fn partitions(n: usize) -> Vec<Vec<Vec<usize>>> {
    vec![
        Vec::new(),
        (0..n).map(|v| vec![v]).collect(),
        (1..n).map(|v| vec![v]).collect(),
        (0..n)
            .collect::<Vec<_>>()
            .chunks(2)
            .map(<[usize]>::to_vec)
            .collect(),
        vec![(0..n).collect()],
    ]
}

/// Every factoring tried on `task`.
fn factorings(task: &Task) -> Vec<Factoring> {
    let explicit = partitions(task.num_variables())
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
