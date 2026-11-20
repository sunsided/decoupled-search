//! Integration tests for decoupled search: cross-checks against explicit-state
//! Dijkstra, fixtures, and error handling.

mod dijkstra;
mod errors;
mod fixtures;
mod heuristic;
mod plan;
mod random;
mod star;
mod unsolvable;

use decoupled_search::{Factoring, HMax, Plan, astar, search};
use sas_plus::Task;

/// Searches `task` under `factoring` with uniform-cost search and A* with
/// `h^max`, checks that both agree and that the plan is valid, and returns
/// its cost; `None` if decoupled search reports the task unsolvable.
fn checked_cost(task: &Task, factoring: &Factoring) -> Option<u64> {
    let uniform = validated(task, search(task, factoring).expect("supported task"));
    let informed = validated(
        task,
        astar(task, factoring, &HMax::new(task)).expect("supported task"),
    );
    assert_eq!(uniform, informed, "A* with h^max under {factoring:?}");
    uniform
}

/// The cost of `plan` after checking it with [`Plan::validate`]; `None` if
/// there is no plan.
fn validated(task: &Task, plan: Option<Plan>) -> Option<u64> {
    let plan = plan?;
    assert_eq!(plan.validate(task), Ok(()), "plan {:?}", plan.ops);
    Some(plan.cost)
}
