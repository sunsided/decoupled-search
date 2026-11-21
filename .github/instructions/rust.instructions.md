---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- The crate is `#![no_std]` with `alloc`, with and without features (the CI `no-std` job).
  `MasHeuristic` sits behind `merge-and-shrink`, `LmCutHeuristic` behind `lm-cut`; flag any
  `std` use and any feature-gated item without its `#[cfg]` on the re-export in `lib.rs`.
- Cost-optimal searches (`search`, `astar`, `anytime_astar`) are checked against explicit
  Dijkstra (`tests/it/dijkstra.rs`, which shares no logic with the crate) under every
  configuration (`optimal_configs` in `tests/it/main.rs`) and factoring strategy, on the
  fixtures, seeded random tasks and the property tests. A new dominance criterion, factoring
  strategy or search option joins those checks in the same PR. Greedy search must find a
  plan exactly when Dijkstra does, and every plan passes `Plan::validate`.
- Admissible heuristics (`HMax`, `MasHeuristic`, `LmCutHeuristic`) never exceed the optimal
  cost and return `None` only for proven dead ends; `tests/it/props.rs` checks that on
  arbitrary explicit states. Unsupported tasks (conditional effects or axioms the decoupled
  semantics cannot express) are rejected with a precise `Error`, never searched wrongly.
- The fuzz targets `search` (factoring and search entry points against explicit search) and
  `plan` (`Plan::validate` against `Task::validate_plan`) cover the same ground with
  arbitrary inputs. A new entry point that builds factorings or searches joins `search`.
- Public API changes go with a matching update to the README scope list, the crate docs in
  `lib.rs` and `docs/design.md`. Name every public API change in the PR.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
