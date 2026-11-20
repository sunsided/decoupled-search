//! The merge-and-shrink adapter: exact on single states with exact
//! bisimulation, and A* with it matches Dijkstra on decoupled states.

use decoupled_search::{DecoupledHeuristic, FactPrices, HMax, MasHeuristic};
use merge_and_shrink::{Bisimulation, LinearMerge, MergeAndShrink, SccDfp};
use sas_plus::random::{RandomTaskParams, random_task};

use crate::checked_cost_with;
use crate::dijkstra::optimal_cost;
use crate::random::factorings;

/// Number of random tasks per test.
const SEEDS: u64 = 100;

/// With exact bisimulation and no size limit, the adapter on the initial
/// state alone is the optimal cost.
#[test]
fn exact_on_the_initial_state() {
    for seed in 0..SEEDS {
        let task = random_task(seed, RandomTaskParams::default());
        let merge = LinearMerge::in_variable_order(task.num_variables());
        let h = MasHeuristic::build(&task, MergeAndShrink::new(merge, Bisimulation));
        let facts = FactPrices::from_state(&task, &task.initial_state());
        assert_eq!(h.estimate(&facts), optimal_cost(&task), "seed {seed}");
    }
}

/// A* with exact and size-limited merge-and-shrink heuristics matches
/// Dijkstra under every factoring and pruning configuration.
#[test]
fn astar_matches_dijkstra() {
    for seed in 0..SEEDS {
        let task = random_task(seed, RandomTaskParams::default());
        let expected = optimal_cost(&task);
        let linear = LinearMerge::in_variable_order(task.num_variables());
        let exact = MasHeuristic::build(&task, MergeAndShrink::new(linear, Bisimulation));
        let limited = MasHeuristic::build(
            &task,
            MergeAndShrink {
                max_states: 6,
                ..MergeAndShrink::new(SccDfp::new(&task), Bisimulation)
            },
        );
        let h_max = HMax::new(&task);
        for factoring in factorings(&task) {
            assert_eq!(
                checked_cost_with(&task, &factoring, &[&exact, &limited, &h_max]),
                expected,
                "seed {seed}, {factoring:?}"
            );
        }
    }
}
