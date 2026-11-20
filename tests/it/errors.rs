//! Rejection of unsupported tasks and malformed factorings.

use decoupled_search::{Error, Factoring, MAX_LEAF_STATES, search};
use sas_plus::{Axiom, Effect, Fact, Operator, State, Task, Variable};

/// Two binary variables and one operator setting both.
fn tiny() -> Task {
    Task {
        variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
        operators: vec![Operator {
            name: "op".into(),
            pre: vec![],
            eff: vec![Effect::new(Fact::new(0, 1)), Effect::new(Fact::new(1, 1))],
            cost: 1,
        }],
        initial: State(vec![0, 0]),
        goal: vec![Fact::new(1, 1)],
        axioms: vec![],
    }
}

/// Conditional effects and axioms the decoupled semantics cannot express,
/// and invalid tasks, are errors, not wrong plans.
#[test]
fn unsupported_tasks_are_rejected() {
    // Effect 0 sets a under b = 0.
    let mut task = tiny();
    task.operators[0].eff[0].conditions.push(Fact::new(1, 0));
    let split = Factoring::new(&task, vec![vec![0], vec![1]]).expect("valid");
    assert_eq!(
        search(&task, &split),
        Err(Error::EffectConditionAcrossLeaves { op: 0, effect: 0 })
    );
    let center_a = Factoring::new(&task, vec![vec![1]]).expect("valid");
    assert_eq!(
        search(&task, &center_a),
        Err(Error::CenterEffectReadsLeaf { op: 0, effect: 0 })
    );
    let one_leaf = Factoring::new(&task, vec![vec![0, 1]]).expect("valid");
    assert!(search(&task, &one_leaf).is_ok());

    // d = 1 iff a = 1 and b = 1.
    let mut task = tiny();
    task.variables.push(Variable {
        axiom_layer: Some(0),
        ..Variable::new("d", 2)
    });
    task.initial.0.push(0);
    task.axioms.push(Axiom {
        conditions: vec![Fact::new(0, 1), Fact::new(1, 1)],
        head: Fact::new(2, 1),
    });
    let split = Factoring::new(&task, vec![vec![0], vec![1]]).expect("valid");
    assert_eq!(
        search(&task, &split),
        Err(Error::AxiomAcrossLeaves {
            var: 2,
            leaves: (0, 1)
        })
    );
    assert_eq!(
        Factoring::new(&task, vec![vec![2]]),
        Err(Error::DerivedInLeaf { var: 2 })
    );
    let attached = Factoring::new(&task, vec![vec![0, 1]]).expect("valid");
    assert!(search(&task, &attached).is_ok());

    let mut task = tiny();
    task.goal[0].value = 5;
    let factoring = Factoring::fork(&task);
    assert!(matches!(
        search(&task, &factoring),
        Err(Error::InvalidTask(_))
    ));
}

/// Each malformed factoring gets its own error.
#[test]
fn malformed_factorings_are_rejected() {
    let task = tiny();
    assert_eq!(
        Factoring::new(&task, vec![vec![]]),
        Err(Error::EmptyLeaf { leaf: 0 })
    );
    assert_eq!(
        Factoring::new(&task, vec![vec![2]]),
        Err(Error::UnknownVariable { var: 2 })
    );
    assert_eq!(
        Factoring::new(&task, vec![vec![0], vec![0]]),
        Err(Error::DuplicateVariable { var: 0 })
    );

    let mut big = tiny();
    big.variables[0].domain_size = u32::try_from(MAX_LEAF_STATES).expect("fits");
    assert_eq!(
        Factoring::new(&big, vec![vec![0, 1]]),
        Err(Error::LeafTooLarge { leaf: 0 })
    );

    let mut wider = tiny();
    wider.variables.push(Variable::new("c", 2));
    wider.initial.0.push(0);
    let factoring = Factoring::new(&task, vec![]).expect("valid");
    assert_eq!(search(&wider, &factoring), Err(Error::FactoringMismatch));
}
