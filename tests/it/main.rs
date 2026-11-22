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
mod props;
mod random;
mod star;
mod stubborn;
#[cfg(feature = "symbolic-leaves")]
mod symbolic;
mod unsolvable;

use decoupled_search::{Config, DecoupledHeuristic, FactPrices, Factoring, HMax, Plan, Pruning};
use sas_plus::Task;

/// Every dominance criterion.
const PRUNINGS: [Pruning; 6] = [
    Pruning::Basic,
    Pruning::Frontier,
    Pruning::EffectivePrices,
    Pruning::Simulation,
    Pruning::EffectiveSimulation,
    Pruning::Adaptive,
];

/// Every configuration of the cost-optimal searches: each dominance
/// criterion with and without the shared `g` budget, and leaf reduction
/// under three of them, each with and without stubborn sets.
fn optimal_configs() -> impl Iterator<Item = Config> {
    let plain = PRUNINGS.into_iter().flat_map(|pruning| {
        [true, false].map(|fold_g| Config {
            pruning,
            fold_g,
            ..Config::default()
        })
    });
    let reduced = [
        Pruning::Basic,
        Pruning::EffectiveSimulation,
        Pruning::Adaptive,
    ]
    .map(|pruning| Config {
        pruning,
        reduce_leaves: true,
        ..Config::default()
    });
    plain.chain(reduced).flat_map(|config| {
        [false, true].map(|stubborn_sets| Config {
            stubborn_sets,
            ..config
        })
    })
}

/// Searches `task` under `factoring` with uniform-cost search and A* with
/// `h^max` (and LM-cut with the `lm-cut` feature), under every
/// configuration of [`optimal_configs`], checks that
/// all agree and that every plan is valid, and returns the cost; `None` if
/// decoupled search reports the task unsolvable.
fn checked_cost(task: &Task, factoring: &Factoring) -> Option<u64> {
    #[cfg(feature = "lm-cut")]
    {
        let lm_cut = decoupled_search::LmCutHeuristic::new(task);
        checked_cost_with(task, factoring, &[&HMax::new(task), &lm_cut])
    }
    #[cfg(not(feature = "lm-cut"))]
    checked_cost_with(task, factoring, &[&HMax::new(task)])
}

/// [`checked_cost`] with A* runs for each of `heuristics` instead of `h^max`,
/// and anytime A* runs with each of them as the fork heuristic, under a
/// blind root heuristic and `h^max` of the task with free leaf actions.
fn checked_cost_with(
    task: &Task,
    factoring: &Factoring,
    heuristics: &[&dyn DecoupledHeuristic],
) -> Option<u64> {
    let free = factoring
        .with_free_leaf_actions(task)
        .expect("supported task");
    let root = HMax::new(&free);
    let roots: [&dyn DecoupledHeuristic; 2] = [&Blind, &root];
    let mut costs = optimal_configs().flat_map(|config| {
        let uniform = config.search(task, factoring).expect("supported task");
        let informed = heuristics
            .iter()
            .map(move |h| config.astar(task, factoring, h).expect("supported task"));
        let anytime = heuristics.iter().flat_map(move |h| {
            roots.into_iter().map(move |r| {
                config
                    .anytime_astar(task, factoring, r, h)
                    .expect("supported task")
            })
        });
        core::iter::once(uniform)
            .chain(informed)
            .chain(anytime)
            .map(move |outcome| (config, validated(task, outcome.plan)))
    });
    let (_, first) = costs.next().expect("at least one configuration");
    for (config, cost) in costs {
        assert_eq!(cost, first, "{config:?} under {factoring:?}");
    }

    first
}

/// The heuristic that is 0 everywhere: blind search, and a trivially
/// root-admissible root heuristic for anytime A*.
struct Blind;

impl DecoupledHeuristic for Blind {
    fn estimate(&self, _: &FactPrices<'_>) -> Option<u64> {
        Some(0)
    }
}

/// The cost of `plan` after checking it with [`Plan::validate`]; `None` if
/// there is no plan.
fn validated(task: &Task, plan: Option<Plan>) -> Option<u64> {
    let plan = plan?;
    assert_eq!(plan.validate(task), Ok(()), "plan {:?}", plan.ops);
    Some(plan.cost)
}
