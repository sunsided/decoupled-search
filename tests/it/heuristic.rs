//! Greedy best-first decoupled search with `h^FF` and `h^max`: valid plans,
//! the same solvability as Dijkstra, and costs no lower than the optimum.

use decoupled_search::{Factoring, HFf, HMax, gbfs};
use sas_plus::Task;
use sas_plus::random::{RandomTaskParams, random_task};

use crate::dijkstra::optimal_cost;
use crate::random::factorings;
use crate::star::{SEEDS, random_star_task, strategies};
use crate::validated;

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
