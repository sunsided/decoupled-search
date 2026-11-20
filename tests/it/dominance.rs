//! The dominance criteria prune what they should: each stronger criterion
//! removes the blow-up of Torralba et al.'s Logistics example, and the
//! shared `g` budget prunes more than the plain `g` comparison.

use decoupled_search::{Config, Factoring, HFf, HMax, Leaves, Pruning, Statistics};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

use crate::dijkstra::optimal_cost;
use crate::random::factorings;
use crate::{PRUNINGS, checked_cost};

/// The running example of Torralba, Gnad, Dubbert and Hoffmann (2016): a
/// truck drives between `n` locations (cost 1 each way), a package starts at
/// location 0 and must reach location 1, loading and unloading cost 1.
/// Package values: `0..n` at a location, `n` in the truck.
fn logistics(n: u32) -> Task {
    let op = |name: String, pre: Vec<Fact>, eff: Fact| Operator {
        name,
        pre,
        eff: vec![Effect::new(eff)],
        cost: 1,
    };
    let mut operators = Vec::new();
    for x in 0..n {
        for y in (0..n).filter(|&y| y != x) {
            operators.push(op(
                format!("drive-{x}-{y}"),
                vec![Fact::new(0, x)],
                Fact::new(0, y),
            ));
        }

        let at = Fact::new(0, x);
        operators.push(op(
            format!("load-{x}"),
            vec![at, Fact::new(1, x)],
            Fact::new(1, n),
        ));
        operators.push(op(
            format!("unload-{x}"),
            vec![at, Fact::new(1, n)],
            Fact::new(1, x),
        ));
    }

    Task {
        variables: vec![Variable::new("truck", n), Variable::new("pkg", n + 1)],
        operators,
        initial: State(vec![0, 0]),
        goal: vec![Fact::new(1, 1)],
        axioms: vec![],
    }
}

/// The statistics of uniform-cost search on `task` under `factoring` with
/// `pruning`, after checking the cost against `expected`.
fn statistics(
    task: &Task,
    factoring: &Factoring,
    pruning: Pruning,
    expected: Option<u64>,
) -> Statistics {
    let config = Config {
        pruning,
        ..Config::default()
    };
    let outcome = config.search(task, factoring).expect("supported");
    assert_eq!(outcome.plan.map(|p| p.cost), expected, "{pruning:?}");
    outcome.statistics
}

/// Basic dominance stores every set of visited locations (exponential in
/// `n`); frontier, effective-price and simulation dominance do not.
#[test]
fn stronger_criteria_avoid_the_logistics_blow_up() {
    let task = logistics(6);
    let factoring = Factoring::fork(&task);
    assert_eq!(factoring.leaves(), &[vec![1]]);
    let expected = optimal_cost(&task);
    assert_eq!(expected, Some(3));
    assert_eq!(checked_cost(&task, &factoring), expected);

    let stored = PRUNINGS.map(|p| statistics(&task, &factoring, p, expected).stored);
    let [basic, frontier, effective, simulation, both, _] = stored;
    assert!(frontier < basic, "{stored:?}");
    assert!(effective <= frontier, "{stored:?}");
    assert!(simulation < basic, "{stored:?}");
    assert!(both <= effective.min(simulation), "{stored:?}");
}

/// The same task, but the goal is unreachable: the search must exhaust the
/// space, where the difference between the criteria shows fully.
#[test]
fn stronger_criteria_shrink_exhaustive_searches() {
    let mut task = logistics(6);
    task.operators.retain(|op| op.name != "unload-1");
    let factoring = Factoring::fork(&task);
    let stored = PRUNINGS.map(|p| statistics(&task, &factoring, p, None).stored);
    let [basic, frontier, effective, simulation, both, _] = stored;
    assert!(basic > 2 * frontier, "{stored:?}");
    assert!(effective <= frontier, "{stored:?}");
    assert!(basic > 2 * simulation, "{stored:?}");
    assert!(both <= effective.min(simulation), "{stored:?}");
}

/// Over random tasks and factorings, the shared `g` budget and each
/// criterion beyond the basic one prune states the weaker ones keep.
#[test]
fn stronger_settings_prune_more_on_random_tasks() {
    let params = RandomTaskParams {
        num_vars: 5,
        num_ops: 14,
        max_cost: 4,
        ..RandomTaskParams::default()
    };
    let config = |pruning, fold_g| Config {
        pruning,
        fold_g,
        ..Config::default()
    };
    let settings = [
        config(Pruning::Basic, false),
        config(Pruning::Basic, true),
        config(Pruning::Frontier, true),
        config(Pruning::EffectivePrices, true),
        config(Pruning::Simulation, true),
        config(Pruning::EffectiveSimulation, true),
    ];
    let mut stored = [0_u64; 6];
    for seed in 0..100 {
        let task = random_task(seed, params);
        for factoring in factorings(&task) {
            for (total, config) in stored.iter_mut().zip(&settings) {
                *total += config
                    .search(&task, &factoring)
                    .expect("supported")
                    .statistics
                    .stored;
            }
        }
    }

    let [plain_g, folded, frontier, effective, simulation, both] = stored;
    assert!(folded < plain_g, "{stored:?}");
    assert!(frontier < folded, "{stored:?}");
    assert!(effective < folded, "{stored:?}");
    assert!(simulation < folded, "{stored:?}");
    assert!(both < folded, "{stored:?}");
}

/// Leaf reduction on the Logistics example: unloading anywhere but the goal
/// and loading at the goal are redundant (the truck simulates the other
/// locations, the goal simulates the truck), so fewer leaf states are
/// reached and fewer decoupled states stored, at the same cost.
#[test]
fn leaf_reduction_shrinks_the_logistics_leaf() {
    let task = logistics(6);
    let factoring = Factoring::fork(&task);
    let run = |reduce_leaves| {
        let config = Config {
            pruning: Pruning::Basic,
            reduce_leaves,
            ..Config::default()
        };
        let outcome = config.search(&task, &factoring).expect("supported");
        assert_eq!(outcome.plan.map(|p| p.cost), Some(3));
        outcome.statistics
    };

    let (full, reduced) = (run(false), run(true));
    assert_eq!(full.reduced_leaf_transitions, 0);
    // 5 non-goal unloads and the load at the goal.
    assert_eq!(reduced.reduced_leaf_transitions, 6);
    assert_eq!(
        reduced.unreachable_leaf_states, 4,
        "only start, truck and goal stay"
    );
    assert!(reduced.stored < full.stored, "{reduced:?} vs {full:?}");
}

/// The adaptive default skips the simulation on quick searches and computes
/// it on long ones (Dijkstra agreement is checked with every other
/// configuration in `checked_cost`).
#[test]
fn adaptive_pruning_simulates_only_when_it_pays() {
    let quick = logistics(3);
    let factoring = Factoring::fork(&quick);
    let stats = statistics(&quick, &factoring, Pruning::Adaptive, Some(3));
    assert_eq!(stats.simulated_leaves, 0, "{stats:?}");

    let params = RandomTaskParams {
        num_vars: 6,
        max_domain: 3,
        num_ops: 16,
        max_cost: 4,
        ..RandomTaskParams::default()
    };
    let (mut skipped, mut simulated) = (0, 0);
    for seed in 0..50 {
        let task = random_task(seed, params);
        let factoring = Factoring::new(&task, vec![vec![0, 1], vec![2, 3]]).expect("valid");
        let config = Config::default();
        let stats = config
            .search(&task, &factoring)
            .expect("supported")
            .statistics;
        if stats.simulated_leaves == 0 {
            skipped += 1;
        } else {
            simulated += 1;
        }
    }

    assert!(
        skipped > 0 && simulated > 0,
        "{skipped} skipped, {simulated} simulated"
    );
}

/// Heuristics on [`Config::reduced_task`]: on the Logistics example only
/// loading at the start and unloading at the goal remain of the package's
/// twelve operators; on random tasks under every factoring, A* with
/// `h^max` of the reduced task under leaf reduction matches Dijkstra.
#[test]
fn heuristics_on_the_reduced_task_stay_admissible() {
    let task = logistics(6);
    let factoring = Factoring::fork(&task);
    let reduce = |pruning| Config {
        pruning,
        reduce_leaves: true,
        ..Config::default()
    };
    let reduced = reduce(Pruning::Basic)
        .reduced_task(&task, &factoring)
        .expect("supported");
    assert_eq!(reduced.operators.len(), task.operators.len() - 10);

    let params = RandomTaskParams {
        num_ops: 12,
        ..RandomTaskParams::default()
    };
    let (mut runs, mut shrunk) = (0, 0);
    for seed in 0..60 {
        let task = random_task(seed, params);
        let expected = optimal_cost(&task);
        for factoring in factorings(&task) {
            for pruning in [
                Pruning::Basic,
                Pruning::EffectiveSimulation,
                Pruning::Adaptive,
            ] {
                let config = reduce(pruning);
                let reduced = config.reduced_task(&task, &factoring).expect("supported");
                shrunk += u64::from(reduced.operators.len() < task.operators.len());
                let outcome = config
                    .astar(&task, &factoring, &HMax::new(&reduced))
                    .expect("supported");
                let plan = outcome
                    .plan
                    .inspect(|p| assert_eq!(p.validate(&task), Ok(())));
                assert_eq!(plan.map(|p| p.cost), expected, "seed {seed}, {factoring:?}");
                runs += 1;
            }
        }
    }

    assert!(
        shrunk > runs / 10,
        "only {shrunk} of {runs} runs removed operators"
    );
}

/// Greedy search with `h^FF` of the reduced task, under leaf reduction and
/// with priced or reachability-only leaves, declares no solvable state a
/// dead end: it is as solvable as Dijkstra says.
#[test]
fn greedy_search_on_the_reduced_task_keeps_solvability() {
    let params = RandomTaskParams {
        num_ops: 12,
        ..RandomTaskParams::default()
    };
    for seed in 0..60 {
        let task = random_task(seed, params);
        let solvable = optimal_cost(&task).is_some();
        for factoring in factorings(&task) {
            for leaves in [Leaves::Priced, Leaves::Reachability] {
                let config = Config {
                    leaves,
                    reduce_leaves: true,
                    ..Config::default()
                };
                let reduced = config.reduced_task(&task, &factoring).expect("supported");
                let plan = config
                    .gbfs(&task, &factoring, &HFf::new(&reduced))
                    .expect("supported")
                    .plan;
                assert_eq!(
                    plan.is_some(),
                    solvable,
                    "seed {seed}, {leaves:?}, {factoring:?}"
                );
                if let Some(plan) = plan {
                    assert_eq!(plan.validate(&task), Ok(()));
                }
            }
        }
    }
}
