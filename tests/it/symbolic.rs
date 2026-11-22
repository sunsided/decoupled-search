//! Symbolic leaves search the same decoupled state space as explicit ones:
//! same costs, valid plans and the same search statistics, on random tasks
//! and on a leaf too large to enumerate.

use decoupled_search::{
    Config, DecoupledHeuristic, Error, Factoring, HFf, HMax, Leaves, Outcome, Pruning,
};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

use crate::dijkstra::optimal_cost;
use crate::random::factorings;

/// The counters both representations must agree on.
fn counters(outcome: &Outcome) -> [u64; 6] {
    let s = outcome.statistics;
    [
        s.expanded,
        s.generated,
        s.dominated,
        s.dead_ends,
        s.stored,
        s.bound_pruned,
    ]
}

/// The plan cost after validating the plan.
fn cost(task: &Task, outcome: &Outcome) -> Option<u64> {
    outcome.plan.as_ref().map(|plan| {
        assert_eq!(plan.validate(task), Ok(()), "plan {:?}", plan.ops);
        plan.cost
    })
}

/// Runs `search` with explicit and with symbolic leaves and checks that
/// costs and counters agree; returns the cost.
///
/// Costs are compared only with priced leaves: with reachability-only
/// leaves, the leaf paths a plan uses are any reached ones, so two valid
/// plans may differ in cost.
fn agree(
    task: &Task,
    what: &str,
    config: Config,
    search: impl Fn(Config) -> Result<Outcome, Error>,
) -> Option<u64> {
    let explicit = search(config).expect("supported");
    let symbolic = search(Config {
        symbolic_leaves: true,
        ..config
    })
    .expect("supported");
    let expected = cost(task, &explicit);
    let found = cost(task, &symbolic);
    if config.leaves == Leaves::Priced {
        assert_eq!(found, expected, "{what}, {config:?}");
    } else {
        assert_eq!(found.is_some(), expected.is_some(), "{what}, {config:?}");
    }
    assert_eq!(
        counters(&symbolic),
        counters(&explicit),
        "{what}, {config:?}"
    );
    expected
}

/// Random tasks under every test factoring: uniform-cost search, A*,
/// anytime A* and GBFS (priced and reachability-only) agree with the
/// explicit search, with and without `fold_g`; optimal costs match
/// Dijkstra.
#[test]
fn symbolic_leaves_match_explicit_leaves() {
    let params = RandomTaskParams {
        num_vars: 5,
        max_domain: 3,
        num_ops: 12,
        ..RandomTaskParams::default()
    };
    for seed in 0..60 {
        let mut task = random_task(seed, params);
        for op in task.operators.iter_mut().step_by(3) {
            op.cost = 0;
        }
        let optimal = optimal_cost(&task);
        let h_max = HMax::new(&task);
        let h_ff = HFf::new(&task);
        for factoring in factorings(&task) {
            for fold_g in [false, true] {
                let config = Config {
                    pruning: Pruning::Basic,
                    fold_g,
                    ..Config::default()
                };
                let f = &factoring;
                let uniform = agree(&task, "uniform", config, |c| c.search(&task, f));
                assert_eq!(uniform, optimal, "seed {seed}");
                let informed = agree(&task, "A*", config, |c| c.astar(&task, f, &h_max));
                assert_eq!(informed, optimal, "seed {seed}");
                let blind: &dyn DecoupledHeuristic = &crate::Blind;
                let anytime = agree(&task, "anytime", config, |c| {
                    c.anytime_astar(&task, f, blind, &h_max)
                });
                assert_eq!(anytime, optimal, "seed {seed}");
                for leaves in [Leaves::Priced, Leaves::Reachability] {
                    let greedy = agree(&task, "GBFS", Config { leaves, ..config }, |c| {
                        c.gbfs(&task, f, &h_ff)
                    });
                    assert_eq!(greedy.is_some(), optimal.is_some(), "seed {seed}");
                }
            }
        }
    }
}

/// A single leaf of 20 binary variables (about a million leaf states,
/// beyond `MAX_LEAF_STATES`): a center switch enables setting the leaf
/// bits one by one. The explicit path rejects the leaf; the symbolic path
/// finds the plan.
#[test]
fn a_leaf_too_large_to_enumerate_is_searched() {
    let n = 20;
    let mut operators = vec![Operator {
        name: "switch".into(),
        pre: vec![Fact::new(0, 0)],
        eff: vec![Effect::new(Fact::new(0, 1))],
        cost: 1,
    }];
    for i in 1..=n {
        operators.push(Operator {
            name: format!("set{i}"),
            pre: vec![Fact::new(0, 1), Fact::new(i, 0)],
            eff: vec![Effect::new(Fact::new(i, 1))],
            cost: 2,
        });
    }
    let task = Task {
        variables: (0..=n).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
        operators,
        initial: State(vec![0; n + 1]),
        goal: (1..=n).map(|i| Fact::new(i, 1)).collect(),
        axioms: vec![],
    };
    let factoring = Factoring::new(&task, vec![(1..=n).collect()]).expect("valid");

    let explicit = Config {
        pruning: Pruning::Basic,
        ..Config::default()
    };
    assert_eq!(
        explicit.search(&task, &factoring),
        Err(Error::LeafTooLarge { leaf: 0 })
    );

    let symbolic = Config {
        symbolic_leaves: true,
        ..explicit
    };
    let outcome = symbolic.search(&task, &factoring).expect("supported");
    let plan = outcome.plan.expect("solvable");
    assert_eq!(plan.validate(&task), Ok(()));
    assert_eq!(plan.cost, 1 + 2 * 20);
}

/// Options that analyse explicit leaf states are rejected.
#[test]
fn unsupported_options_are_rejected() {
    let task = random_task(0, RandomTaskParams::default());
    let factoring = Factoring::fork(&task);
    let rejected = |config: Config| {
        matches!(
            Config {
                symbolic_leaves: true,
                ..config
            }
            .search(&task, &factoring),
            Err(Error::SymbolicUnsupported { .. })
        )
    };
    assert!(rejected(Config::default()), "adaptive pruning");
    let basic = Config {
        pruning: Pruning::Basic,
        ..Config::default()
    };
    assert!(rejected(Config {
        reduce_leaves: true,
        ..basic
    }));
    assert!(rejected(Config {
        stubborn_sets: true,
        ..basic
    }));
}
