# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/sunsided/decoupled-search/compare/v0.2.0...v0.3.0) - 2026-11-22

### Changed

- Skip leaf closure where a global action changes nothing the leaf reads
- Build enumerated symbolic leaves on the compiled leaf
- Check issue #3's acceptance on tasks without a star factoring
- Check every heuristic under a symbolic center and non-star factorings
- Mix enumerated and symbolic leaves, and search with a symbolic center
- Find generalized factorings with directly interacting leaves
- Keep leaf pricing functions as BDDs with symbolic leaves
- Depend on planning-heuristics 0.2
- Propose symbolic leaf representation in the design notes

### Fixed

- Fix factorings that split derived variables from their inputs
- Fix h^FF reporting a reachable goal as a dead end on deep tasks

### Removed

- Remove dead code in leaf simulation and the relaxation

## [0.2.0](https://github.com/sunsided/decoupled-search/compare/v0.1.0...v0.2.0) - 2026-11-22

### Changed

- Measure stubborn set pruning on random tasks and under GBFS
- Prune decoupled expansions with strong stubborn sets
- Specify decoupled strong stubborn sets in the design notes

## [0.1.0](https://github.com/sunsided/decoupled-search/releases/tag/v0.1.0) - 2026-11-22

### Added

- Add the full lint table, clippy.toml and rustfmt.toml
- Add dormant release automation with release-plz
- Add CONTRIBUTING.md
- Add git hooks and Copilot review instructions
- Add README badges, usage and features sections and absolute links
- Add property tests for the factoring strategies and admissibility
- Add search and plan fuzz targets
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos configuration
- Add the Codecov configuration and ignore the coverage report
- Add the Taskfile and CI workflows
- Add an LM-cut heuristic on decoupled states behind the lm-cut feature
- Add sample-search factoring selection with an expansion budget
- Add anytime decoupled A* (Anytime Fork-Root A*)
- Add the X-shape factoring strategy
- Implement the Beyond Forks factoring strategies and abstain in auto
- Add simulation-based leaf reduction and an adaptive default pruning
- Add a merge-and-shrink heuristic adapter behind a feature
- Add frontier, effective-price and simulation dominance, and Config
- Support conditional effects and axioms where decoupling allows
- Add Plan::validate and a public FactPrices constructor
- Add a search bench and document the implemented scope
- Add decoupled A* and GBFS with h^max and h^FF on fact prices
- Add inverted-fork and greedy star factorings with a star check
- Implement uniform-cost decoupled search over fork factorings
- Add the design document for decoupled search

### Changed

- Describe the release-plz gate as a skip condition
- Version the crate as 0.1.0
- Depend on planning-heuristics from crates.io
- Report dependencies and workflow findings to GitHub code security
- Describe the released scope and audience in the README
- Raise the MSRV to 1.86 to match planning-heuristics
- Describe the property tests and fuzz targets in the testing strategy
- Build the property tests without the lm-cut feature
- Separate the let-else guard in the search fuzz target
- Render mermaid diagrams on docs.rs
- Depend on merge-and-shrink from crates.io
- Bench seeded LM-cut against the price-level rebuild
- Evaluate LM-cut on decoupled states from seeded fact costs
- Bench A* with LM-cut on gripper, on the full and the reduced task
- Check reduced-task heuristics under greedy search and update the docs
- Bench anytime A* on gripper
- Keep the inputs of derived variables in one factor in the star strategies
- Align leaf reduction with the technical report's superfluous transitions
- Put the dominance and merge-and-shrink proofs on public items
- Document the new scope in the design, README and crate docs
- Score factorings and choose a strategy with Factoring::auto
- Store leaf prices sparsely in decoupled states
- Test fork shapes and zero-cost operators, and fix design references
- Cross-check decoupled search against explicit-state Dijkstra
- Reserve the crate name with a 0.0.0 placeholder

### Fixed

- Fix star and leaf-size docs, and run blind A*/GBFS on the loop task

### Removed

- Drop pre-release wording from the contributor and design docs
- Drop operators made useless by leaf reduction for heuristics
