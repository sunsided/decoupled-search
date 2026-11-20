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

/// Times uniform-cost search, A* with `h^max` and GBFS with `h^FF` under
/// the fork factoring.
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
    ] {
        let config = Config {
            pruning,
            ..Config::default()
        };
        criterion.bench_function(&format!("uniform-{name}/fork"), |b| {
            b.iter(|| config.search(black_box(&task), &factoring));
        });
    }
}

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_search(&mut criterion);
    bench_pruning(&mut criterion);
    criterion.final_summary();
}
