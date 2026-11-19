//! Integration tests for decoupled search: cross-checks against explicit-state
//! Dijkstra, fixtures, and error handling.

mod dijkstra;
mod errors;
mod fixtures;
mod random;
mod unsolvable;

use decoupled_search::{Factoring, search};
use sas_plus::Task;

/// Searches `task` under `factoring`, checks the plan against the task, and
/// returns its cost; `None` if decoupled search reports the task unsolvable.
fn checked_cost(task: &Task, factoring: &Factoring) -> Option<u64> {
    let plan = search(task, factoring).expect("supported task")?;
    assert_eq!(
        task.validate_plan(&plan.ops),
        Some(plan.cost),
        "plan {:?} is invalid or its cost is wrong under {factoring:?}",
        plan.ops
    );
    Some(plan.cost)
}
