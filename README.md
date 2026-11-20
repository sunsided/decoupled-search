# decoupled-search

Decoupled search for factored SAS+ planning: center and leaf factoring, leaf reachability and leaf-state dominance.

## Status

Pre-release. The API will change. Implemented (see
[docs/design.md](docs/design.md)):

- Factorings: fork, inverted fork, greedy incident-arcs and maximum-leaves
  star factorings, explicit partitions, and a strict-star check.
  `Factoring::score` reports leaves, mobile leaves, the leaf-action share
  and leaf sizes; `Factoring::auto` picks the best-scored strategy.
- Leaf reachability with prices (cheapest cost per leaf state along the
  center path); stored states keep only reached leaf states, as `u16` ranks
  with `u64` prices (no prices at all when all are 0).
- Uniform-cost search and A* (cost-optimal), and GBFS (satisficing), with
  `Config` choosing the dominance criterion: basic, frontier, effective
  prices, leaf simulation, or effective prices over simulation (Torralba,
  Gnad, Dubbert and Hoffmann 2016, generalised to global actions that touch
  leaves), optionally with one `g` budget shared by all leaves. Search
  statistics come with every `Outcome`.
- Reachability-only leaves (no prices) for GBFS.
- Conditional effects on a leaf conditioned on the center and that leaf,
  on the center conditioned on the center; derived variables that depend on
  the center and at most one leaf. Everything else is rejected with a
  precise error.
- Heuristics on decoupled states: `h^max` (admissible) and `h^FF`, both
  seeded with per-fact leaf prices (`FactPrices`, constructible for tests),
  and with the `merge-and-shrink` feature an admissible merge-and-shrink
  adapter (`MasHeuristic`).
- Reconstruction of a sequential plan for the original task, and
  `Plan::validate`.

```rust
use decoupled_search::{Config, Factoring, HMax, Pruning, astar, search};

let factoring = Factoring::auto(&task);
let plan = search(&task, &factoring)?;
let plan = astar(&task, &factoring, &HMax::new(&task))?;
plan.expect("solvable").validate(&task)?;

let outcome = Config { pruning: Pruning::Basic, ..Config::default() }.search(&task, &factoring)?;
println!("{:?}", outcome.statistics);
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
