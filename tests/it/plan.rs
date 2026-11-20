//! `Plan::validate` on hand-built plans, including conditional effects and
//! axioms, and against `Task::validate_plan` on random plans.

use decoupled_search::{Plan, PlanError};
use sas_plus::random::{RandomTaskParams, Rng, random_task};
use sas_plus::{Axiom, Effect, Fact, Operator, State, Task, Variable};

/// `x` toggles from 0 to 1 at cost 2; with `x = 1`, `flip` sets `y = 1`
/// only if `x = 1` (a conditional effect). Derived `d` holds iff `y = 1`,
/// and the goal is `d = 1`.
fn conditional() -> Task {
    Task {
        variables: vec![
            Variable::new("x", 2),
            Variable::new("y", 2),
            Variable {
                axiom_layer: Some(0),
                ..Variable::new("d", 2)
            },
        ],
        operators: vec![
            Operator {
                name: "set-x".into(),
                pre: vec![Fact::new(0, 0)],
                eff: vec![Effect::new(Fact::new(0, 1))],
                cost: 2,
            },
            Operator {
                name: "flip".into(),
                pre: vec![],
                eff: vec![Effect {
                    conditions: vec![Fact::new(0, 1)],
                    fact: Fact::new(1, 1),
                }],
                cost: 1,
            },
        ],
        initial: State(vec![0, 0, 0]),
        goal: vec![Fact::new(2, 1)],
        axioms: vec![Axiom {
            conditions: vec![Fact::new(1, 1)],
            head: Fact::new(2, 1),
        }],
    }
}

/// Every error variant, each for its own reason.
#[test]
fn validate_reports_each_error() {
    let task = conditional();
    let plan = |ops: Vec<usize>, cost| Plan { ops, cost };

    assert_eq!(plan(vec![0, 1], 3).validate(&task), Ok(()));
    assert_eq!(
        plan(vec![1], 1).validate(&task),
        Err(PlanError::GoalNotReached),
        "the conditional effect does not fire while x = 0"
    );
    assert_eq!(
        plan(vec![0, 0], 4).validate(&task),
        Err(PlanError::NotApplicable { step: 1, op: 0 })
    );
    assert_eq!(
        plan(vec![0, 7], 3).validate(&task),
        Err(PlanError::UnknownOperator { step: 1, op: 7 })
    );
    assert_eq!(
        plan(vec![0, 1], 4).validate(&task),
        Err(PlanError::CostMismatch {
            claimed: 4,
            actual: 3
        })
    );

    let mut invalid = task;
    invalid.goal[0].value = 9;
    assert!(matches!(
        plan(vec![], 0).validate(&invalid),
        Err(PlanError::InvalidTask(_))
    ));
}

/// On random operator sequences of random tasks with conditional effects and
/// axioms, `validate` accepts exactly what `Task::validate_plan` accepts.
#[test]
fn validate_agrees_with_sas_plus() {
    let params = RandomTaskParams {
        max_conditions: 2,
        num_derived: 2,
        ..RandomTaskParams::default()
    };
    let mut accepted = 0;
    for seed in 0..300 {
        let task = random_task(seed, params);
        let mut rng = Rng::new(seed);
        let len = rng.range(0, 4);
        let ops: Vec<usize> = (0..len)
            .map(|_| {
                let max = u64::try_from(task.operators.len() - 1).expect("small");
                usize::try_from(rng.range(0, max)).expect("small")
            })
            .collect();
        let expected = task.validate_plan(&ops);
        let cost = expected.unwrap_or(0);
        let result = Plan { ops, cost }.validate(&task);
        assert_eq!(
            result.is_ok(),
            expected.is_some(),
            "seed {seed}: {result:?}"
        );
        accepted += u64::from(result.is_ok());
    }

    assert!(accepted > 5, "only {accepted} accepted plans");
}
