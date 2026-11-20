//! Greedy best-first decoupled search with `h^FF` and `h^max`: valid plans,
//! the same solvability as Dijkstra, and costs no lower than the optimum.

use decoupled_search::{Config, Error, Factoring, HFf, HMax, Leaves, Pruning, gbfs};
use sas_plus::Task;
use sas_plus::random::{RandomTaskParams, random_task};

use crate::dijkstra::optimal_cost;
use crate::random::factorings;
use crate::star::{SEEDS, random_star_task, strategies};
use crate::{PRUNINGS, validated};

/// Checks greedy search under `factoring` with both heuristics against the
/// optimal cost `expected`, and returns how many runs found a plan.
fn check_greedy(task: &Task, factoring: &Factoring, expected: Option<u64>) -> u64 {
    let ff = validated(
        task,
        gbfs(task, factoring, &HFf::new(task)).expect("supported"),
    );
    let max = validated(
        task,
        gbfs(task, factoring, &HMax::new(task)).expect("supported"),
    );
    for cost in [ff, max] {
        assert_eq!(
            cost.is_some(),
            expected.is_some(),
            "solvability under {factoring:?}"
        );
        assert!(
            cost >= expected,
            "{cost:?} beats the optimum {expected:?} under {factoring:?}"
        );
    }

    u64::from(ff.is_some()) + u64::from(max.is_some())
}

/// Greedy search on star tasks under every automatic strategy.
#[test]
fn greedy_search_is_sound_on_star_tasks() {
    let mut solved = 0;
    for seed in 0..SEEDS {
        let task = random_star_task(seed);
        let expected = optimal_cost(&task);
        for (_, factoring) in strategies(&task) {
            solved += check_greedy(&task, &factoring, expected);
        }
    }

    assert!(solved > SEEDS, "only {solved} solved runs");
}

/// Greedy search on unstructured random tasks under explicit partitions.
#[test]
fn greedy_search_is_sound_on_random_tasks() {
    let mut solved = 0;
    for seed in 0..SEEDS {
        let task = random_task(seed, RandomTaskParams::default());
        let expected = optimal_cost(&task);
        for factoring in factorings(&task) {
            solved += check_greedy(&task, &factoring, expected);
        }
    }

    assert!(solved > SEEDS, "only {solved} solved runs");
}

/// Greedy search with reachability-only leaves under every dominance
/// criterion: valid plans exactly when Dijkstra finds one, never cheaper
/// than the optimum, and the cost is the plan's operator cost.
#[test]
fn reachability_only_gbfs_matches_dijkstra_solvability() {
    let random = (0..SEEDS).map(|seed| random_task(seed, RandomTaskParams::default()));
    let mut solved = 0;
    for task in (0..SEEDS).map(random_star_task).chain(random) {
        let expected = optimal_cost(&task);
        let ff = HFf::new(&task);
        for factoring in factorings(&task) {
            for (pruning, reduce_leaves) in PRUNINGS
                .map(|p| (p, false))
                .into_iter()
                .chain([(Pruning::Basic, true)])
            {
                let config = Config {
                    pruning,
                    leaves: Leaves::Reachability,
                    reduce_leaves,
                    ..Config::default()
                };
                let outcome = config.gbfs(&task, &factoring, &ff).expect("supported");
                let cost = validated(&task, outcome.plan);
                assert_eq!(
                    cost.is_some(),
                    expected.is_some(),
                    "{config:?}, {factoring:?}"
                );
                assert!(cost >= expected, "{cost:?} beats {expected:?}");
                solved += u64::from(cost.is_some());
            }
        }
    }

    assert!(solved > SEEDS, "only {solved} solved runs");
}

/// Reachability-only leaves are refused for cost-optimal search.
#[test]
fn reachability_only_leaves_need_greedy_search() {
    let task = random_star_task(0);
    let factoring = Factoring::fork(&task);
    let config = Config {
        leaves: Leaves::Reachability,
        ..Config::default()
    };
    assert_eq!(
        config.search(&task, &factoring),
        Err(Error::ReachabilityNotOptimal)
    );
    assert_eq!(
        config.astar(&task, &factoring, &HMax::new(&task)),
        Err(Error::ReachabilityNotOptimal)
    );
}
