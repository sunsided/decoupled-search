//! Hand-encoded gripper fixtures with known optimal costs.

use decoupled_search::Factoring;
use sas_plus::fdr::SasFile;
use sas_plus::{Axiom, Effect, Fact, Operator, State, Task, Variable};

use crate::checked_cost;
use crate::dijkstra::optimal_cost;

/// The task in `tests/fixtures/<name>/output.sas`.
fn fixture(name: &str) -> Task {
    let path = format!(
        "{}/tests/fixtures/{name}/output.sas",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture exists");
    SasFile::parse(&text).expect("fixture parses").to_task()
}

/// Fork and ball-leaf factorings find the known optima.
#[test]
fn gripper_optimal_costs() {
    for (name, optimum) in [("gripper-costs", 7), ("gripper-unit", 5)] {
        let task = fixture(name);
        assert_eq!(optimal_cost(&task), Some(optimum), "{name}");

        let fork = Factoring::fork(&task);
        assert_eq!(fork.center(), &[0], "{name}: the robot is the center");
        assert_eq!(checked_cost(&task, &fork), Some(optimum), "{name}");

        // Balls as leaves: picks and drops also change the grippers, so they become global.
        let balls = Factoring::new(&task, vec![vec![1], vec![2]]).expect("valid");
        assert_eq!(checked_cost(&task, &balls), Some(optimum), "{name}");
    }
}

/// A truck (variable 0) and two packages that each ride along: drive costs 2,
/// load and unload 1.
fn trucks_and_packages() -> Task {
    let op = |name: &str, pre: Vec<Fact>, eff: Fact, cost| Operator {
        name: name.into(),
        pre,
        eff: vec![Effect::new(eff)],
        cost,
    };
    let mut operators = vec![
        op("drive-ab", vec![Fact::new(0, 0)], Fact::new(0, 1), 2),
        op("drive-ba", vec![Fact::new(0, 1)], Fact::new(0, 0), 2),
    ];
    // Package values: 0 at a, 1 at b, 2 in the truck.
    for pkg in [1, 2] {
        for (place, name) in [(0, "a"), (1, "b")] {
            let at = Fact::new(0, place);
            operators.push(op(
                &format!("load-{pkg}-{name}"),
                vec![at, Fact::new(pkg, place)],
                Fact::new(pkg, 2),
                1,
            ));
            operators.push(op(
                &format!("unload-{pkg}-{name}"),
                vec![at, Fact::new(pkg, 2)],
                Fact::new(pkg, place),
                1,
            ));
        }
    }

    Task {
        variables: vec![
            Variable::new("truck", 2),
            Variable::new("p1", 3),
            Variable::new("p2", 3),
        ],
        operators,
        initial: State(vec![0, 0, 1]),
        goal: vec![Fact::new(1, 1), Fact::new(2, 0)],
        axioms: vec![],
    }
}

/// The fork factoring of a logistics-like task has the truck as center and
/// one leaf per package.
#[test]
fn fork_factoring_splits_off_packages() {
    let task = trucks_and_packages();
    let fork = Factoring::fork(&task);
    assert_eq!(fork.center(), &[0]);
    assert_eq!(fork.leaves(), &[vec![1], vec![2]]);
    assert_eq!(optimal_cost(&task), Some(8));
    assert_eq!(checked_cost(&task, &fork), Some(8));
}

/// A truck (center, at `a` or `b`), a package leaf (at `a`, at `b`, or in
/// the truck) and a flag leaf. Unloading puts the package wherever the
/// truck is (two center-conditioned effects on the leaf), and derived
/// `done` holds once the package is at `b` and the truck back at `a`, so it
/// is attached to the package leaf. Optimal: load, drive, unload, set the
/// flag at `b` (cost 2), drive back: 6.
fn delivery() -> Task {
    let op = |name: &str, pre: Vec<Fact>, eff: Vec<Effect>, cost| Operator {
        name: name.into(),
        pre,
        eff,
        cost,
    };
    let conditional = |truck, pkg| Effect {
        conditions: vec![Fact::new(0, truck)],
        fact: Fact::new(1, pkg),
    };
    Task {
        variables: vec![
            Variable::new("truck", 2),
            Variable::new("pkg", 3),
            Variable::new("flag", 2),
            Variable {
                axiom_layer: Some(0),
                ..Variable::new("done", 2)
            },
        ],
        operators: vec![
            op(
                "drive-ab",
                vec![Fact::new(0, 0)],
                vec![Effect::new(Fact::new(0, 1))],
                1,
            ),
            op(
                "drive-ba",
                vec![Fact::new(0, 1)],
                vec![Effect::new(Fact::new(0, 0))],
                1,
            ),
            op(
                "load",
                vec![Fact::new(0, 0), Fact::new(1, 0)],
                vec![Effect::new(Fact::new(1, 2))],
                1,
            ),
            op(
                "unload",
                vec![Fact::new(1, 2)],
                vec![conditional(0, 0), conditional(1, 1)],
                1,
            ),
            op(
                "flag",
                vec![Fact::new(0, 1)],
                vec![Effect::new(Fact::new(2, 1))],
                2,
            ),
        ],
        initial: State(vec![0, 0, 0, 0]),
        goal: vec![Fact::new(3, 1), Fact::new(2, 1)],
        axioms: vec![Axiom {
            conditions: vec![Fact::new(1, 1), Fact::new(0, 0)],
            head: Fact::new(3, 1),
        }],
    }
}

/// Conditional effects on a leaf and a derived variable attached to it, on
/// a hand-checked task, under the fork factoring and an explicit one.
#[test]
fn delivery_with_conditional_effects_and_axioms() {
    let task = delivery();
    assert_eq!(optimal_cost(&task), Some(6));

    let explicit = Factoring::new(&task, vec![vec![1], vec![2]]).expect("valid");
    assert_eq!(checked_cost(&task, &explicit), Some(6));

    // The package feeds the derived variable, so it is no sink; the derived
    // variable itself never becomes a leaf variable.
    let fork = Factoring::fork(&task);
    assert_eq!(fork.leaves(), &[vec![2]]);
    assert_eq!(fork.center(), &[0, 1, 3]);
    assert_eq!(checked_cost(&task, &fork), Some(6));

    let auto = Factoring::auto(&task);
    assert_eq!(checked_cost(&task, &auto), Some(6), "{auto:?}");
}
