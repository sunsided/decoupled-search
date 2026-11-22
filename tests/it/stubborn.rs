//! Decoupled strong stubborn sets keep optimal costs on the examples of Gnad,
//! Hoffmann and Wehrle (JAIR 2019) and prune what the paper says they prune.
//! The randomized cross-checks run with them through [`crate::optimal_configs`].

use decoupled_search::{Config, Factoring, HMax, Pruning, Statistics};
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

use crate::dijkstra::optimal_cost;

/// An operator with unconditional effects.
fn op(name: String, pre: Vec<Fact>, eff: Vec<Fact>, cost: u32) -> Operator {
    Operator {
        name,
        pre,
        eff: eff.into_iter().map(Effect::new).collect(),
        cost,
    }
}

/// The cost and statistics of uniform-cost search with or without stubborn
/// sets, after checking the plan.
fn run(task: &Task, factoring: &Factoring, stubborn_sets: bool) -> (Option<u64>, Statistics) {
    let config = Config {
        stubborn_sets,
        ..Config::default()
    };
    let outcome = config.search(task, factoring).expect("supported");
    let cost = outcome.plan.map(|plan| {
        assert_eq!(plan.validate(task), Ok(()));
        plan.cost
    });
    (cost, outcome.statistics)
}

/// Example 2 of the paper: the initial state is already a goal state (walk,
/// price 100), but a manager job (cost 1) unlocks a free company car and a
/// drive at cost 1. The goal-price frontier must keep `get-manager-job`.
#[test]
fn the_company_car_is_found_in_a_goal_state() {
    // job: worker 0, manager 1; car: no 0, yes 1; location: A 0, B 1.
    let task = Task {
        variables: vec![
            Variable::new("job", 2),
            Variable::new("have-car", 2),
            Variable::new("location", 2),
        ],
        operators: vec![
            op(
                "get-manager-job".into(),
                vec![Fact::new(0, 0)],
                vec![Fact::new(0, 1)],
                1,
            ),
            op(
                "walk".into(),
                vec![Fact::new(2, 0)],
                vec![Fact::new(2, 1)],
                100,
            ),
            op(
                "drive".into(),
                vec![Fact::new(1, 1), Fact::new(2, 0)],
                vec![Fact::new(2, 1)],
                1,
            ),
            op(
                "get-company-car".into(),
                vec![Fact::new(0, 1), Fact::new(1, 0)],
                vec![Fact::new(1, 1)],
                0,
            ),
            op(
                "buy-car".into(),
                vec![Fact::new(1, 0)],
                vec![Fact::new(1, 1)],
                20000,
            ),
        ],
        initial: State(vec![0, 0, 0]),
        goal: vec![Fact::new(2, 1)],
        axioms: vec![],
    };
    let factoring = Factoring::new(&task, vec![vec![1, 2]]).expect("valid");

    assert_eq!(run(&task, &factoring, true).0, Some(2));
    let config = Config {
        stubborn_sets: true,
        ..Config::default()
    };
    let outcome = config
        .astar(&task, &factoring, &HMax::new(&task))
        .expect("supported");
    assert_eq!(outcome.plan.map(|p| p.cost), Some(2));
}

/// The paper's transport example: one truck (A 0, B 1) and `n` packages (A 0,
/// B 1, in the truck 2), all at A, to be brought to B. In the `NoEmpty`
/// variant the truck only drives with some package inside, which makes the
/// factoring a star rather than a fork.
fn transport(n: u32, no_empty: bool) -> Task {
    let mut operators = Vec::new();
    for x in 0..2 {
        let y = 1 - x;
        if no_empty {
            for p in 1..=n as usize {
                operators.push(op(
                    format!("drive-{x}-{y}-p{p}"),
                    vec![Fact::new(0, x), Fact::new(p, 2)],
                    vec![Fact::new(0, y)],
                    1,
                ));
            }
        } else {
            operators.push(op(
                format!("drive-{x}-{y}"),
                vec![Fact::new(0, x)],
                vec![Fact::new(0, y)],
                1,
            ));
        }
        for p in 1..=n as usize {
            operators.push(op(
                format!("load-p{p}-{x}"),
                vec![Fact::new(0, x), Fact::new(p, x)],
                vec![Fact::new(p, 2)],
                1,
            ));
            operators.push(op(
                format!("unload-p{p}-{x}"),
                vec![Fact::new(0, x), Fact::new(p, 2)],
                vec![Fact::new(p, x)],
                1,
            ));
        }
    }

    let n = n as usize;
    Task {
        variables: core::iter::once(Variable::new("truck", 2))
            .chain((1..=n).map(|p| Variable::new(format!("p{p}"), 3)))
            .collect(),
        operators,
        initial: State(vec![0; n + 1]),
        goal: (1..=n).map(|p| Fact::new(p, 1)).collect(),
        axioms: vec![],
    }
}

/// Both transport variants keep their optimal cost under every dominance
/// criterion.
#[test]
fn transport_keeps_its_optimal_cost() {
    for no_empty in [false, true] {
        let task = transport(3, no_empty);
        let factoring = Factoring::new(&task, (1..=3).map(|p| vec![p]).collect()).expect("valid");
        let expected = optimal_cost(&task);
        for pruning in crate::PRUNINGS {
            for fold_g in [false, true] {
                let config = Config {
                    pruning,
                    fold_g,
                    stubborn_sets: true,
                    ..Config::default()
                };
                let outcome = config.search(&task, &factoring).expect("supported");
                assert_eq!(
                    outcome.plan.map(|p| p.cost),
                    expected,
                    "{pruning:?}, fold_g {fold_g}, no_empty {no_empty}"
                );
            }
        }
    }
}

/// The first family of Theorem 5: `m` trucks, each with its own `n`
/// packages that only it can carry. All trucks are in the center, so plain
/// decoupled search stores exponentially many states in `m`; with stubborn
/// sets the search commits to one truck at a time.
fn own_packages(m: usize, n: usize) -> Task {
    let mut variables: Vec<Variable> = (0..m).map(|t| Variable::new(format!("t{t}"), 2)).collect();
    let mut operators = Vec::new();
    for t in 0..m {
        for x in 0..2 {
            operators.push(op(
                format!("drive-t{t}-{x}"),
                vec![Fact::new(t, x)],
                vec![Fact::new(t, 1 - x)],
                1,
            ));
        }
    }
    let mut goal = Vec::new();
    for t in 0..m {
        for _ in 0..n {
            let p = variables.len();
            variables.push(Variable::new(format!("p{p}"), 3));
            goal.push(Fact::new(p, 1));
            for x in 0..2 {
                operators.push(op(
                    format!("load-p{p}-{x}"),
                    vec![Fact::new(t, x), Fact::new(p, x)],
                    vec![Fact::new(p, 2)],
                    1,
                ));
                operators.push(op(
                    format!("unload-p{p}-{x}"),
                    vec![Fact::new(t, x), Fact::new(p, 2)],
                    vec![Fact::new(p, x)],
                    1,
                ));
            }
        }
    }

    let count = variables.len();
    Task {
        variables,
        operators,
        initial: State(vec![0; count]),
        goal,
        axioms: vec![],
    }
}

/// Stubborn sets remove the blow-up in the number of trucks: `m + 1` stored
/// states, as the paper states, against exponentially many in `m` without
/// them, at the same cost.
#[test]
fn one_truck_at_a_time() {
    let (m, n) = (6, 2);
    let task = own_packages(m, n);
    let factoring =
        Factoring::new(&task, (m..m + m * n).map(|p| vec![p]).collect()).expect("valid");
    let (plain_cost, plain) = run(&task, &factoring, false);
    let (cost, pruned) = run(&task, &factoring, true);

    assert_eq!(cost, plain_cost);
    assert_eq!(cost, Some(u64::try_from(m * (2 * n + 1)).expect("small")));
    assert!(pruned.stubborn_pruned > 0);
    assert!(plain.stored >= 1 << m, "{plain:?}");
    assert_eq!(pruned.stored, m as u64 + 1, "{pruned:?}");
}

/// Under every dominance criterion the pruned search stays optimal on the
/// truck family, and never stores more than the plain one.
#[test]
fn one_truck_at_a_time_under_every_criterion() {
    let task = own_packages(3, 2);
    let factoring = Factoring::new(&task, (3..9).map(|p| vec![p]).collect()).expect("valid");
    for pruning in [Pruning::Basic, Pruning::Frontier, Pruning::Adaptive] {
        let config = |stubborn_sets| Config {
            pruning,
            stubborn_sets,
            ..Config::default()
        };
        let plain = config(false).search(&task, &factoring).expect("supported");
        let pruned = config(true).search(&task, &factoring).expect("supported");
        assert_eq!(
            pruned.plan.map(|p| p.cost),
            plain.plan.map(|p| p.cost),
            "{pruning:?}"
        );
        assert!(
            pruned.statistics.stored <= plain.statistics.stored,
            "{pruning:?}"
        );
    }
}
