//! Integration tests for decoupled search: cross-checks against explicit-state
//! Dijkstra, fixtures, and error handling.

mod dijkstra;
mod dominance;
mod errors;
mod fixtures;
mod heuristic;
#[cfg(feature = "merge-and-shrink")]
mod mas;
mod plan;
mod random;
mod star;
mod unsolvable;

use decoupled_search::{Config, DecoupledHeuristic, Factoring, HMax, Plan, Pruning};
use sas_plus::Task;

/// Every dominance criterion.
const PRUNINGS: [Pruning; 5] = [
    Pruning::Basic,
    Pruning::Frontier,
    Pruning::EffectivePrices,
    Pruning::Simulation,
    Pruning::EffectiveSimulation,
];

/// Every configuration of the cost-optimal searches: each dominance
/// criterion, with and without the shared `g` budget.
fn optimal_configs() -> impl Iterator<Item = Config> {
    PRUNINGS.into_iter().flat_map(|pruning| {
        [true, false].map(|fold_g| Config {
            pruning,
            fold_g,
            ..Config::default()
        })
    })
}

/// Searches `task` under `factoring` with uniform-cost search and A* with
/// `h^max`, under every configuration of [`optimal_configs`], checks that
/// all agree and that every plan is valid, and returns the cost; `None` if
/// decoupled search reports the task unsolvable.
fn checked_cost(task: &Task, factoring: &Factoring) -> Option<u64> {
    checked_cost_with(task, factoring, &[&HMax::new(task)])
}

/// [`checked_cost`] with A* runs for each of `heuristics` instead of `h^max`.
fn checked_cost_with(
    task: &Task,
    factoring: &Factoring,
    heuristics: &[&dyn DecoupledHeuristic],
) -> Option<u64> {
    let mut costs = optimal_configs().flat_map(|config| {
        let uniform = config.search(task, factoring).expect("supported task");
        let informed = heuristics
            .iter()
            .map(move |h| config.astar(task, factoring, h).expect("supported task"));
        core::iter::once(uniform)
            .chain(informed)
            .map(move |outcome| (config, validated(task, outcome.plan)))
    });
    let (_, first) = costs.next().expect("at least one configuration");
    for (config, cost) in costs {
        assert_eq!(cost, first, "{config:?} under {factoring:?}");
    }

    first
}

/// The cost of `plan` after checking it with [`Plan::validate`]; `None` if
/// there is no plan.
fn validated(task: &Task, plan: Option<Plan>) -> Option<u64> {
    let plan = plan?;
    assert_eq!(plan.validate(task), Ok(()), "plan {:?}", plan.ops);
    Some(plan.cost)
}
