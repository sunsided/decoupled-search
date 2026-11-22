# decoupled-search

[![crates.io](https://img.shields.io/crates/v/decoupled-search.svg)](https://crates.io/crates/decoupled-search)
[![docs.rs](https://img.shields.io/docsrs/decoupled-search)](https://docs.rs/decoupled-search)
[![CI](https://github.com/sunsided/decoupled-search/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/decoupled-search/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/decoupled-search/graph/badge.svg)](https://codecov.io/gh/sunsided/decoupled-search)
[![MSRV](https://img.shields.io/crates/msrv/decoupled-search)](https://crates.io/crates/decoupled-search)
[![License](https://img.shields.io/crates/l/decoupled-search)](https://github.com/sunsided/decoupled-search#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Decoupled (star-topology) search for factored planning tasks in
[sas-plus](https://github.com/sunsided/sas-plus)'s SAS+ representation, after Gnad and Hoffmann (ICAPS 2015,
AIJ 2018). A factoring splits the variables into a center and several leaves; the search then follows paths of
global actions only and keeps, per leaf, the cheapest price of every leaf state reachable along the current path.
On tasks with loosely coupled components (logistics-style packages around a shared vehicle, independent machines
around a common resource), that removes the interleavings of leaf moves which blow up an explicit state space, and
the search still returns an ordinary sequential plan for the original task.

The crate covers the factoring strategies, cost-optimal (uniform-cost, A*, anytime A*) and satisficing (GBFS)
search, the dominance pruning of Torralba, Gnad, Dubbert and Hoffmann (2016) and decoupled heuristics, with plan reconstruction and
an independent plan validator. It is `no_std` with `alloc`, with and without the optional features
except `symbolic-leaves`, which needs `std`.

## Who this is for

- Planning researchers working on star-topology decoupling, factoring selection or dominance pruning who want a
  readable implementation, cross-checked against explicit search, to build on or compare against.
- Planner authors whose tasks fall apart into a center and many weakly interacting leaves (transport, multi-robot
  delivery, manufacturing cells), where explicit search spends most of its effort on interleavings of leaf moves.
- Embedded and robotics teams that plan onboard: the crate builds for `no_std` targets.
- Tasks without a useful star factoring gain nothing from decoupling; `Factoring::auto` then falls back to explicit
  search, and for explicit-state search on its own
  [heuristic-search](https://github.com/sunsided/heuristic-search) is the better fit.
- For the heuristics themselves on ordinary states, use
  [planning-heuristics](https://github.com/sunsided/planning-heuristics); for symbolic search over the same tasks,
  [symbolic-search](https://github.com/sunsided/symbolic-search).

## Installation

```toml
[dependencies]
decoupled-search = "0.1"
```

## Scope

More detail, including how each rule maps to the papers, is in
[docs/design.md](https://github.com/sunsided/decoupled-search/blob/main/docs/design.md).

- Factorings: fork, inverted fork, X-shape, and the incident-arcs and
  independent-set (maximum leaves) strict-star strategies of Gnad, Poser
  and Hoffmann (IJCAI 2017), the generalized factorings of Gnad, Torralba
  and Fišer (ICAPS 2022) with directly interacting leaves and a possibly
  empty center, explicit partitions, and a strict-star check.
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
- Optional partial-order reduction (`Config::stubborn_sets`): the
  decoupled strong stubborn sets of Gnad, Hoffmann and Wehrle (JAIR 2019)
  for general star factorings. The paper proves they keep optimality and
  completeness under duplicate detection; the combinations with dominance
  pruning and leaf reduction that its proof does not cover are listed in
  the `stubborn_sets` documentation.
- Optional symbolic leaves (`Config::symbolic_leaves`, feature `symbolic-leaves`): leaf
  pricing functions as BDDs (Gnad, Torralba and Hoffmann, SoCS 2017) for leaves too large
  to enumerate, with small leaves kept enumerated next to them. Same state space and results
  as explicit leaves, under basic dominance. Optionally a symbolic center
  (`Config::symbolic_center`): one search node per set of center states that share their
  leaf pricing and cost.
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
  [planning-heuristics](https://crates.io/crates/planning-heuristics).

- `symbolic-leaves`: `Config::symbolic_leaves`, leaf pricing functions as BDDs, and
  `Config::symbolic_center`, search nodes that hold sets of center states, both on
  [symbolic-search](https://crates.io/crates/symbolic-search). Needs `std`.

The crate is `no_std` with `alloc`, with and without the features, except `symbolic-leaves`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/decoupled-search/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/decoupled-search/blob/main/LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/decoupled-search/blob/main/CONTRIBUTING.md).
