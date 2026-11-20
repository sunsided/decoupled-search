# decoupled-search

Decoupled search for factored SAS+ planning: center and leaf factoring, leaf reachability and leaf-state dominance.

## Status

Pre-release. The API will change. Implemented (see
[docs/design.md](docs/design.md)):

- Factorings: fork, inverted fork, greedy incident-arcs and maximum-leaves
  star factorings, explicit partitions, and a strict-star check.
- Leaf reachability with prices (cheapest cost per leaf state along the
  center path); stored states keep only reached leaf states, as `u16` ranks
  with `u64` prices.
- Uniform-cost search and A* (cost-optimal), and GBFS (satisficing), with
  leaf-state dominance pruning.
- Heuristics on decoupled states: `h^max` (admissible) and `h^FF`, both
  seeded with per-fact leaf prices.
- Reconstruction of a sequential plan for the original task.

Not supported yet: axioms, conditional effects, the stronger dominance
criteria of Torralba et al. (2016), and `merge-and-shrink` heuristics.

```rust
use decoupled_search::{Factoring, HMax, astar, search};

let factoring = Factoring::incident_arcs(&task);
let plan = search(&task, &factoring)?;
let plan = astar(&task, &factoring, &HMax::new(&task))?;
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
