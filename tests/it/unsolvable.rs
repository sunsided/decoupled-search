//! A task whose prices grow forever along a center loop, so only dominance
//! pruning makes the search terminate.

use decoupled_search::Factoring;
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

use crate::checked_cost;
use crate::dijkstra::optimal_cost;

/// An operator with one effect.
fn op(name: &str, pre: Vec<Fact>, eff: Fact) -> Operator {
    Operator {
        name: name.into(),
        pre,
        eff: vec![Effect::new(eff)],
        cost: 1,
    }
}

/// Center `c` toggles only while leaf `x` matches it, and `x` toggles only
/// while `c` differs; each round raises the leaf prices. Goal `y = 1` is
/// never set.
fn alternating() -> Task {
    let var = |name: &str| Variable::new(name, 2);
    Task {
        variables: vec![var("c"), var("x"), var("y")],
        operators: vec![
            op(
                "c-up",
                vec![Fact::new(0, 0), Fact::new(1, 0)],
                Fact::new(0, 1),
            ),
            op(
                "c-down",
                vec![Fact::new(0, 1), Fact::new(1, 1)],
                Fact::new(0, 0),
            ),
            op(
                "x-up",
                vec![Fact::new(0, 1), Fact::new(1, 0)],
                Fact::new(1, 1),
            ),
            op(
                "x-down",
                vec![Fact::new(0, 0), Fact::new(1, 1)],
                Fact::new(1, 0),
            ),
        ],
        initial: State(vec![0, 0, 0]),
        goal: vec![Fact::new(2, 1)],
        axioms: vec![],
    }
}

#[test]
fn growing_prices_terminate_on_unsolvable_task() {
    let task = alternating();
    assert_eq!(optimal_cost(&task), None);

    let factoring = Factoring::new(&task, vec![vec![1]]).expect("valid");
    assert_eq!(checked_cost(&task, &factoring), None);
}

#[test]
fn growing_prices_still_find_a_goal_on_the_loop() {
    let mut task = alternating();
    task.goal = vec![Fact::new(0, 0), Fact::new(1, 1)];
    task.operators.push(op("y-up", vec![], Fact::new(2, 1)));

    let factoring = Factoring::new(&task, vec![vec![1]]).expect("valid");
    assert_eq!(checked_cost(&task, &factoring), optimal_cost(&task));
}
