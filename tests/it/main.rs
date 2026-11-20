//! Integration tests for decoupled search: cross-checks against explicit-state
//! Dijkstra, fixtures, and error handling.

mod dijkstra;
mod errors;
mod fixtures;
mod heuristic;
mod random;
mod star;
mod unsolvable;

use decoupled_search::{Factoring, HMax, Plan, astar, search};
use sas_plus::{Fact, OpId, Task};

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

/// The cost of `plan` after checking it with [`plan_cost`]; `None` if there
/// is no plan.
fn validated(task: &Task, plan: Option<Plan>) -> Option<u64> {
    let plan = plan?;
    assert_eq!(
        plan_cost(task, &plan.ops),
        Some(plan.cost),
        "plan {:?} is invalid or its cost is wrong",
        plan.ops
    );
    Some(plan.cost)
}

/// An independent plan validator: applies the operators of `plan` to the
/// initial state fact by fact and returns the plan cost if every operator is
/// applicable and the goal holds at the end. Supports only tasks without
/// axioms and conditional effects.
fn plan_cost(task: &Task, plan: &[OpId]) -> Option<u64> {
    let mut values = task.initial.0.clone();
    let holds = |values: &[u32], facts: &[Fact]| facts.iter().all(|f| values[f.var] == f.value);
    let mut cost = 0;
    for &op in plan {
        let operator = task.operators.get(op)?;
        if !holds(&values, &operator.pre) {
            return None;
        }

        for effect in &operator.eff {
            values[effect.fact.var] = effect.fact.value;
        }

        cost += u64::from(operator.cost);
    }

    holds(&values, &task.goal).then_some(cost)
}
