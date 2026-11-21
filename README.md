# decoupled-search

[![crates.io](https://img.shields.io/crates/v/decoupled-search.svg)](https://crates.io/crates/decoupled-search)
[![docs.rs](https://img.shields.io/docsrs/decoupled-search)](https://docs.rs/decoupled-search)
[![CI](https://github.com/sunsided/decoupled-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/decoupled-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/decoupled-search/graph/badge.svg)](https://codecov.io/gh/sunsided/decoupled-search)
[![MSRV](https://img.shields.io/crates/msrv/decoupled-search)](https://crates.io/crates/decoupled-search)
[![License](https://img.shields.io/crates/l/decoupled-search)](https://github.com/sunsided/decoupled-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Decoupled search for factored SAS+ planning: center and leaf factoring, leaf reachability and leaf-state dominance.

## Status

Pre-release. The API will change. Implemented (see
[docs/design.md](https://github.com/sunsided/decoupled-search/blob/main/docs/design.md)):

- Factorings: fork, inverted fork, X-shape, and the incident-arcs and
  independent-set (maximum leaves) strict-star strategies of Gnad, Poser
  and Hoffmann (IJCAI 2017), explicit partitions, and a strict-star check.
  `Factoring::score` reports leaves, mobile leaves, the leaf-action share
  and leaf sizes; `Factoring::auto` picks the best-scored strategy with at
  least two mobile leaves and otherwise falls back to explicit search;
  `Factoring::auto_by_sample` picks by short sample searches instead
  (the paper's selection, with an expansion budget instead of a clock).
- Leaf reachability with prices (cheapest cost per leaf state along the
  center path); stored states keep only reached leaf states, as `u16` ranks
  with `u64` prices (no prices at all when all are 0).
- Uniform-cost search, A* and anytime A* (Gnad and Hoffmann's Anytime
  Fork-Root A*, cost-optimal), and GBFS (satisficing), with
  `Config` choosing the dominance criterion: basic, frontier, effective
  prices, leaf simulation, or effective prices over simulation (Torralba,
  Gnad, Dubbert and Hoffmann 2016, generalised to global actions that touch
  leaves), optionally with one `g` budget shared by all leaves; the
  default switches to simulation only once the search has spent its cost.
  Optional simulation-based leaf reduction (`Config::reduce_leaves`, the
  superfluous-transition rule of the paper's technical report with a
  removal order that keeps it sound), and `Config::reduced_task` without
  the operators it makes useless, for heuristics.
  Search statistics come with every `Outcome`.
- Reachability-only leaves (no prices) for GBFS.
- Conditional effects on a leaf conditioned on the center and that leaf,
  on the center conditioned on the center; derived variables that depend on
  the center and at most one leaf. Everything else is rejected with a
  precise error.
- Heuristics on decoupled states: `h^max` (admissible) and `h^FF`, both
  seeded with per-fact leaf prices (`FactPrices`, constructible for tests),
  with the `merge-and-shrink` feature an admissible merge-and-shrink
  adapter (`MasHeuristic`), and with the `lm-cut` feature an admissible
  LM-cut adapter (`LmCutHeuristic`, from `planning-heuristics`).
- Reconstruction of a sequential plan for the original task, and
  `Plan::validate`.

## Usage

```rust
use decoupled_search::{Config, Factoring, HMax, Pruning, astar, search};
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

// Two binary variables: `go` sets `c = 1`, then `set` needs `c = 1` and
// sets the goal `x = 1`.
let op = |name: &str, pre: Vec<Fact>, eff: Fact, cost| Operator {
    name: name.into(),
    pre,
    eff: vec![Effect::new(eff)],
    cost,
};
let task = Task {
    variables: vec![Variable::new("c", 2), Variable::new("x", 2)],
    operators: vec![
        op("go", vec![Fact::new(0, 0)], Fact::new(0, 1), 2),
        op("set", vec![Fact::new(0, 1), Fact::new(1, 0)], Fact::new(1, 1), 1),
    ],
    initial: State(vec![0, 0]),
    goal: vec![Fact::new(1, 1)],
    axioms: vec![],
};

let factoring = Factoring::auto(&task);
let plan = search(&task, &factoring)?.expect("solvable");
assert_eq!(plan.validate(&task), Ok(()));
assert_eq!(plan.cost, 3);

let plan = astar(&task, &factoring, &HMax::new(&task))?;
assert_eq!(plan.map(|plan| plan.cost), Some(3));

let outcome = Config { pruning: Pruning::Basic, ..Config::default() }.search(&task, &factoring)?;
println!("{:?}", outcome.statistics);
# Ok::<(), decoupled_search::Error>(())
```

## Features

- `merge-and-shrink`: `MasHeuristic`, an admissible merge-and-shrink heuristic on decoupled
  states, from [merge-and-shrink](https://crates.io/crates/merge-and-shrink).
- `lm-cut`: `LmCutHeuristic`, an admissible LM-cut heuristic on decoupled states, from
  [planning-heuristics](https://github.com/sunsided/planning-heuristics) (not on crates.io
  yet).

The crate is `no_std` with `alloc`, with and without the features.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/decoupled-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/decoupled-search/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/decoupled-search/blob/main/CONTRIBUTING.md),
including how to build against local checkouts of the sibling crates.
