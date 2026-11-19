# decoupled-search

Decoupled search for factored SAS+ planning: center and leaf factoring, leaf reachability and leaf-state dominance.

## Status

Pre-release (milestone 1 of [docs/design.md](docs/design.md)): fork and
explicit center/leaf factorings over `sas-plus` tasks, decoupled states with
per-leaf prices, uniform-cost decoupled search with simple dominance pruning,
and reconstruction of a cost-optimal sequential plan. Axioms and conditional
effects are not supported yet. The API will change.

```rust
use decoupled_search::{Factoring, search};

let plan = search(&task, &Factoring::fork(&task))?;
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
