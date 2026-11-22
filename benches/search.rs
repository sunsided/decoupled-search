//! Benchmarks for decoupled search on the hand-encoded gripper fixture.
//!
//! The fork factoring makes the robot the center and every ball a leaf, so
//! these runs exercise leaf reachability, sparse price storage and
//! dominance checks under every criterion.

use std::hint::black_box;

use criterion::Criterion;
use decoupled_search::{Config, Factoring, HFf, HMax, Leaves, Pruning, astar, gbfs, search};
use sas_plus::Task;
use sas_plus::fdr::SasFile;

/// Reads the `gripper-costs` fixture.
fn gripper() -> Task {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gripper-costs/output.sas"
    );
    let text = std::fs::read_to_string(path).expect("fixture is readable");

    SasFile::parse(&text).expect("fixture parses").to_task()
}

/// Times uniform-cost search, A* and anytime A* with `h^max`, and GBFS with
/// `h^FF` under the fork factoring.
fn bench_search(criterion: &mut Criterion) {
    let task = gripper();
    let factoring = Factoring::fork(&task);
    let (h_max, h_ff) = (HMax::new(&task), HFf::new(&task));

    criterion.bench_function("uniform/fork", |b| {
        b.iter(|| search(black_box(&task), &factoring));
    });
    criterion.bench_function("astar-hmax/fork", |b| {
        b.iter(|| astar(black_box(&task), &factoring, &h_max));
    });
    let root = HMax::new(&factoring.with_free_leaf_actions(&task).expect("supported"));
    let config = Config::default();
    criterion.bench_function("anytime-astar-hmax/fork", |b| {
        b.iter(|| config.anytime_astar(black_box(&task), &factoring, &root, &h_max));
    });
    criterion.bench_function("gbfs-hff/fork", |b| {
        b.iter(|| gbfs(black_box(&task), &factoring, &h_ff));
    });
    let reachability = Config {
        leaves: Leaves::Reachability,
        ..Config::default()
    };
    criterion.bench_function("gbfs-hff-reachability/fork", |b| {
        b.iter(|| reachability.gbfs(black_box(&task), &factoring, &h_ff));
    });
}

/// Times uniform-cost search under every dominance criterion, which
/// exercises the dominance checks and their per-leaf analyses.
fn bench_pruning(criterion: &mut Criterion) {
    let task = gripper();
    let factoring = Factoring::fork(&task);
    for (name, pruning) in [
        ("basic", Pruning::Basic),
        ("frontier", Pruning::Frontier),
        ("effective", Pruning::EffectivePrices),
        ("simulation", Pruning::Simulation),
        ("effective-simulation", Pruning::EffectiveSimulation),
        ("adaptive", Pruning::Adaptive),
    ] {
        for reduce_leaves in [false, true] {
            let config = Config {
                pruning,
                reduce_leaves,
                ..Config::default()
            };
            let suffix = if reduce_leaves { "+reduced" } else { "" };
            criterion.bench_function(&format!("uniform-{name}{suffix}/fork"), |b| {
                b.iter(|| config.search(black_box(&task), &factoring));
            });
        }
    }
}

/// Times A* with an exact merge-and-shrink heuristic under the fork
/// factoring; building the abstraction is not timed.
#[cfg(feature = "merge-and-shrink")]
fn bench_merge_and_shrink(criterion: &mut Criterion) {
    use decoupled_search::MasHeuristic;
    use merge_and_shrink::{Bisimulation, MergeAndShrink, SccDfp};

    let task = gripper();
    let factoring = Factoring::fork(&task);
    let h = MasHeuristic::build(&task, MergeAndShrink::new(SccDfp::new(&task), Bisimulation));
    criterion.bench_function("astar-mas/fork", |b| {
        b.iter(|| astar(black_box(&task), &factoring, &h));
    });
}

/// Times A* with the LM-cut adapter under the fork factoring: seeded and
/// with price levels (rebuilt per estimate) on the full task, and seeded on
/// the task reduced by leaf reduction (with leaf reduction in the search).
#[cfg(feature = "lm-cut")]
fn bench_lm_cut(criterion: &mut Criterion) {
    use decoupled_search::LmCutHeuristic;

    let task = gripper();
    let factoring = Factoring::fork(&task);
    let h = LmCutHeuristic::new(&task);
    criterion.bench_function("astar-lmcut/fork", |b| {
        b.iter(|| astar(black_box(&task), &factoring, &h));
    });
    let levels = LmCutHeuristic::with_price_levels(&task);
    criterion.bench_function("astar-lmcut-levels/fork", |b| {
        b.iter(|| astar(black_box(&task), &factoring, &levels));
    });

    let config = Config {
        reduce_leaves: true,
        ..Config::default()
    };
    let reduced = config.reduced_task(&task, &factoring).expect("supported");
    let h = LmCutHeuristic::new(&reduced);
    criterion.bench_function("astar-lmcut-reduced/fork", |b| {
        b.iter(|| config.astar(black_box(&task), &factoring, &h));
    });
}

/// A center switch and one leaf of `n` binary variables that the switch
/// lets the plan set one by one: `2^n` leaf states.
#[cfg(feature = "symbolic-leaves")]
fn bits(n: usize) -> Task {
    use sas_plus::{Effect, Fact, Operator, State, Variable};

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
    Task {
        variables: (0..=n).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
        operators,
        initial: State(vec![0; n + 1]),
        goal: (1..=n).map(|i| Fact::new(i, 1)).collect(),
        axioms: vec![],
    }
}

/// Times uniform-cost search with explicit and with symbolic leaves under
/// basic dominance: on the gripper fork, where every leaf is tiny, and on
/// one leaf of growing size, where explicit leaves stop at
/// `MAX_LEAF_STATES`.
#[cfg(feature = "symbolic-leaves")]
fn bench_symbolic(criterion: &mut Criterion) {
    let explicit = Config {
        pruning: Pruning::Basic,
        ..Config::default()
    };
    let symbolic = Config {
        symbolic_leaves: true,
        symbolic_min_states: 0,
        ..explicit
    };
    let mixed = Config {
        symbolic_leaves: true,
        ..explicit
    };

    let task = gripper();
    let factoring = Factoring::fork(&task);
    // The explicit counterpart is `uniform-basic/fork` in `bench_pruning`.
    criterion.bench_function("uniform-basic-symbolic/fork", |b| {
        b.iter(|| symbolic.search(black_box(&task), &factoring));
    });
    criterion.bench_function("uniform-basic-mixed/fork", |b| {
        b.iter(|| mixed.search(black_box(&task), &factoring));
    });

    for n in [8, 12, 16, 24] {
        let task = bits(n);
        let factoring = Factoring::new(&task, vec![(1..=n).collect()]).expect("valid");
        if n <= 16 {
            criterion.bench_function(&format!("uniform-basic/bits-{n}"), |b| {
                b.iter(|| explicit.search(black_box(&task), &factoring));
            });
        }
        criterion.bench_function(&format!("uniform-basic-symbolic/bits-{n}"), |b| {
            b.iter(|| symbolic.search(black_box(&task), &factoring));
        });
    }
}

/// `n` center switches that each turn on at cost 1 and one leaf that steps
/// on its own: `2^n` center states, `n + 1` pricing-and-`g` groups.
#[cfg(feature = "symbolic-leaves")]
fn switches(n: usize) -> Task {
    use sas_plus::{Effect, Fact, Operator, State, Variable};

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
    Task {
        variables: (0..=n).map(|i| Variable::new(format!("v{i}"), 2)).collect(),
        operators,
        initial: State(vec![0; n + 1]),
        goal: (0..=n).map(|i| Fact::new(i, 1)).collect(),
        axioms: vec![],
    }
}

/// Times uniform-cost search with an explicit and with a symbolic center
/// on [`switches`], and the symbolic center on the gripper fork, where few
/// center states share a pricing function.
#[cfg(feature = "symbolic-leaves")]
fn bench_symbolic_center(criterion: &mut Criterion) {
    let explicit = Config {
        pruning: Pruning::Basic,
        fold_g: false,
        ..Config::default()
    };
    let grouped = Config {
        symbolic_center: true,
        ..explicit
    };

    for n in [6, 10, 14] {
        let task = switches(n);
        let factoring = Factoring::new(&task, vec![vec![n]]).expect("valid");
        criterion.bench_function(&format!("uniform-basic-nofold/switches-{n}"), |b| {
            b.iter(|| explicit.search(black_box(&task), &factoring));
        });
        criterion.bench_function(&format!("uniform-basic-center/switches-{n}"), |b| {
            b.iter(|| grouped.search(black_box(&task), &factoring));
        });
    }

    let task = gripper();
    let factoring = Factoring::fork(&task);
    criterion.bench_function("uniform-basic-nofold/fork", |b| {
        b.iter(|| explicit.search(black_box(&task), &factoring));
    });
    criterion.bench_function("uniform-basic-center/fork", |b| {
        b.iter(|| grouped.search(black_box(&task), &factoring));
    });
}

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_search(&mut criterion);
    bench_pruning(&mut criterion);
    #[cfg(feature = "merge-and-shrink")]
    bench_merge_and_shrink(&mut criterion);
    #[cfg(feature = "lm-cut")]
    bench_lm_cut(&mut criterion);
    #[cfg(feature = "symbolic-leaves")]
    bench_symbolic(&mut criterion);
    #[cfg(feature = "symbolic-leaves")]
    bench_symbolic_center(&mut criterion);
    criterion.final_summary();
}
