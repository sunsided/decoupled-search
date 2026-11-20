//! The dominance criteria prune what they should: each stronger criterion
//! removes the blow-up of Torralba et al.'s Logistics example, and the
//! shared `g` budget prunes more than the plain `g` comparison.

use decoupled_search::{Config, Factoring, Pruning, Statistics};
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
    let [basic, frontier, effective, simulation, both] = stored;
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
    let [basic, frontier, effective, simulation, both] = stored;
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
