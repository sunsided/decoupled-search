//! Symbolic leaves search the same decoupled state space as explicit ones:
//! same costs, valid plans and the same search statistics, on random tasks
//! (all leaves symbolic, mixed, or all enumerated) and on a leaf too large
//! to enumerate. A symbolic center keeps optimal costs and merges states.

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
    (config, config_symbolic): (Config, impl Fn(Config) -> Config),
    search: impl Fn(Config) -> Result<Outcome, Error>,
) -> Option<u64> {
    let explicit = search(config).expect("supported");
    let symbolic = search(config_symbolic(config)).expect("supported");
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
        for (factoring, min_states) in factorings(&task).into_iter().flat_map(|f| {
            // All leaves symbolic, single-variable leaves enumerated, all
            // leaves enumerated inside the symbolic search.
            [0, 4, usize::MAX].map(|m| (f.clone(), m))
        }) {
            let symbolic = move |c: Config| Config {
                symbolic_leaves: true,
                symbolic_min_states: min_states,
                ..c
            };
            for fold_g in [false, true] {
                let config = Config {
                    pruning: Pruning::Basic,
                    fold_g,
                    ..Config::default()
                };
                let f = &factoring;
                let uniform = agree(&task, "uniform", (config, symbolic), |c| c.search(&task, f));
                assert_eq!(uniform, optimal, "seed {seed}");
                let informed = agree(&task, "A*", (config, symbolic), |c| {
                    c.astar(&task, f, &h_max)
                });
                assert_eq!(informed, optimal, "seed {seed}");
                let blind: &dyn DecoupledHeuristic = &crate::Blind;
                let anytime = agree(&task, "anytime", (config, symbolic), |c| {
                    c.anytime_astar(&task, f, blind, &h_max)
                });
                assert_eq!(anytime, optimal, "seed {seed}");
                for leaves in [Leaves::Priced, Leaves::Reachability] {
                    let greedy = agree(
                        &task,
                        "GBFS",
                        (Config { leaves, ..config }, symbolic),
                        |c| c.gbfs(&task, f, &h_ff),
                    );
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
    let center = Config {
        symbolic_center: true,
        fold_g: false,
        ..basic
    };
    assert!(
        rejected(Config {
            fold_g: true,
            ..center
        }),
        "fold_g"
    );
    let anytime = center.anytime_astar(&task, &factoring, &crate::Blind, &crate::Blind);
    assert!(
        matches!(anytime, Err(Error::SymbolicUnsupported { .. })),
        "anytime A*"
    );
}

/// With a symbolic center, uniform-cost search and A* find the Dijkstra
/// cost and GBFS finds a plan exactly on solvable tasks, under every test
/// factoring and with symbolic or enumerated leaves; every plan validates.
#[test]
fn symbolic_center_keeps_optimal_costs() {
    let mut grouped_differs = 0;
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
        let h_ff = HFf::new(&task);
        for factoring in factorings(&task) {
            for (symbolic_leaves, leaves) in [
                (true, Leaves::Priced),
                (false, Leaves::Priced),
                (true, Leaves::Reachability),
            ] {
                let config = Config {
                    pruning: Pruning::Basic,
                    fold_g: false,
                    symbolic_center: true,
                    symbolic_leaves,
                    symbolic_min_states: 0,
                    leaves,
                    ..Config::default()
                };
                let what = format!("seed {seed}, {config:?}, {factoring:?}");
                let greedy = config.gbfs(&task, &factoring, &h_ff).expect("supported");
                assert_eq!(
                    cost(&task, &greedy).is_some(),
                    optimal.is_some(),
                    "GBFS {what}"
                );
                if leaves == Leaves::Reachability {
                    continue;
                }
                let uniform = config.search(&task, &factoring).expect("supported");
                assert_eq!(cost(&task, &uniform), optimal, "uniform {what}");
                for (name, h) in heuristics(&task) {
                    let informed = config.astar(&task, &factoring, &*h).expect("supported");
                    assert_eq!(cost(&task, &informed), optimal, "A* {name} {what}");
                }
                let concrete = Config {
                    symbolic_center: false,
                    ..config
                };
                let explicit = concrete.search(&task, &factoring).expect("supported");
                if explicit.statistics.stored != uniform.statistics.stored {
                    grouped_differs += 1;
                }
            }
        }
    }

    assert!(
        grouped_differs > 0,
        "the symbolic center never stored a different number of nodes"
    );
}

/// Every admissible heuristic of the crate, by name: center facts that
/// several members of a symbolic-center node have all arrive at price 0.
fn heuristics(task: &Task) -> Vec<(&'static str, Box<dyn DecoupledHeuristic>)> {
    let mut all: Vec<(&'static str, Box<dyn DecoupledHeuristic>)> =
        vec![("h^max", Box::new(HMax::new(task)))];
    #[cfg(feature = "lm-cut")]
    {
        use decoupled_search::LmCutHeuristic;
        all.push(("LM-cut", Box::new(LmCutHeuristic::new(task))));
        all.push((
            "LM-cut levels",
            Box::new(LmCutHeuristic::with_price_levels(task)),
        ));
    }
    #[cfg(feature = "merge-and-shrink")]
    {
        use decoupled_search::MasHeuristic;
        use merge_and_shrink::{Bisimulation, MergeAndShrink, SccDfp};
        let mas = MergeAndShrink::new(SccDfp::new(task), Bisimulation);
        all.push(("merge-and-shrink", Box::new(MasHeuristic::build(task, mas))));
    }

    all
}

/// `n` center switches that each turn on at cost 1 and one leaf that steps
/// on its own: every set of `k` switches on shares one pricing function at
/// `g = k`, so a symbolic center stores `n + 1` nodes where the explicit
/// center stores every subset.
#[test]
fn a_symbolic_center_merges_states_with_one_pricing() {
    let n = 8;
    let mut operators: Vec<Operator> = (0..n)
        .map(|i| Operator {
            name: format!("on{i}"),
            pre: vec![Fact::new(i, 0)],
            eff: vec![Effect::new(Fact::new(i, 1))],
            cost: 1,
        })
        .collect();
    operators.push(Operator {
        name: "step".into(),
        pre: vec![Fact::new(n, 0)],
        eff: vec![Effect::new(Fact::new(n, 1))],
        cost: 1,
    });
    let task = Task {
        variables: (0..=n).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
        operators,
        initial: State(vec![0; n + 1]),
        goal: (0..=n).map(|i| Fact::new(i, 1)).collect(),
        axioms: vec![],
    };
    let factoring = Factoring::new(&task, vec![vec![n]]).expect("valid");
    let base = Config {
        pruning: Pruning::Basic,
        fold_g: false,
        ..Config::default()
    };
    let explicit = base.search(&task, &factoring).expect("supported");
    let grouped = Config {
        symbolic_center: true,
        ..base
    }
    .search(&task, &factoring)
    .expect("supported");

    let expected = Some(u64::try_from(n).expect("small") + 1);
    assert_eq!(cost(&task, &explicit), expected);
    assert_eq!(cost(&task, &grouped), expected);
    assert_eq!(explicit.statistics.stored, 1 << n);
    assert_eq!(
        grouped.statistics.stored,
        u64::try_from(n).expect("small") + 1
    );
}
