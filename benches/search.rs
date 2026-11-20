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

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_search(&mut criterion);
    bench_pruning(&mut criterion);
    #[cfg(feature = "merge-and-shrink")]
    bench_merge_and_shrink(&mut criterion);
    #[cfg(feature = "lm-cut")]
    bench_lm_cut(&mut criterion);
    criterion.final_summary();
}
