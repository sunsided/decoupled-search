# decoupled-search: design

Decoupled (star-topology) search for factored SAS+ planning tasks from
[`sas-plus`](https://crates.io/crates/sas-plus). The crate partitions a
task's variables into one *center* and several *leaves*, searches over center
paths only, and keeps, per leaf, the set of leaf states reachable along the
current center path together with their cheapest cost (the *price*).

Status: milestone 1 (see [Milestones](#milestones)).

## Terminology

| Term | Meaning |
| --- | --- |
| Factoring | A partition of the state variables into a center `C` and leaves `L1..Lk`. `C` may be empty, `k` may be 0. |
| Leaf state | An assignment to the variables of one leaf, stored as a mixed-radix rank. |
| Leaf action of `L` | An operator whose effects all lie in `L` and whose preconditions all lie in `C ∪ L`. |
| Global action | Every other operator: it changes the center, or touches two leaves, or reads another leaf. |
| Decoupled state | A center state plus, per leaf, a price for every leaf state (`∞` = unreached). |
| `g` | The cost of the global actions on the path to a decoupled state. |
| Price | The cost of the cheapest leaf-action sequence that reaches a leaf state, interleaved with the current global path. |

## Factoring strategies

The classification above is our own formulation. It makes every variable
partition a valid factoring: an operator that would break the star topology
(for example one that reads leaf `L1` and writes leaf `L2`) becomes a global
action, so it is searched explicitly instead of being priced. The factoring
therefore only affects performance, never soundness or optimality. Gnad and
Hoffmann (2018) define star factorings by restricting how operators may touch
leaves; we have not cross-checked our rule against their exact definition.

Strategies:

1. **Fork (sink components)**, milestone 1, `Factoring::fork`. Every sink
   strongly connected component of the causal graph (no arcs leaving it) whose
   leaf state space has at most `2^16` states becomes a leaf; everything else
   is the center. Operators on a sink component read only the center and that
   component, so all of them are leaf actions. On a task whose causal graph is
   one component, this yields no leaves and the search degenerates to plain
   uniform-cost search over the center.
2. **Explicit**, milestone 1, `Factoring::new(task, leaves)`. The caller names
   the leaves; the center is the rest. Used by tests to exercise arbitrary
   partitions.
3. **Strict-star factorings**, milestone 3. Gnad, Poser and Hoffmann (2017)
   detect general strict-star topologies, among them a greedy strategy that
   moves highly connected variables (by incident causal-graph arcs) into the
   center; port those after reading the paper (details here are not yet
   checked against it). Score a factoring by number of
   leaves, leaf state-space sizes, and the share of operators that are leaf
   actions; consider rejecting factorings with fewer than two leaves.
4. **Inverted fork / mobile leaves**, later. Only if benchmarks call for it.

## Decoupled state and pricing

```text
DecoupledState {
    center: Vec<Value>,      // values of the center variables, in Factoring::center order
    prices: Vec<Vec<u64>>,   // prices[leaf][rank]; u64::MAX = unreached
}
```

Leaf states are ranked in mixed radix over the leaf's variables, so a leaf
with variables of domain sizes `d0, d1, ...` has `d0 * d1 * ...` ranks.
Reading or setting one variable of a rank is a division, a modulus and an
add; nothing allocates per leaf state.

*Leaf closure.* Given a center state and seed prices for a leaf, Dijkstra over
the leaf's ranks with the leaf actions whose center preconditions hold. The
result is the new price vector. Seeds are all finite-priced ranks.

Prices are dense vectors. That is simple and fast for small leaves, but costs
`O(|S_L|)` memory per leaf per decoupled state. Sparse maps are milestone 2.

## Successor generation

A global action `a` is applicable in decoupled state `s` iff its center
preconditions hold in `s.center` and, for each leaf where it has a
precondition, at least one reached leaf state satisfies it. The successor:

1. applies `a`'s center effects;
2. for each leaf `a` reads or writes: keeps the reached ranks satisfying
   `pre_L(a)`, maps each through `eff_L(a)`, and keeps the minimum price on
   collisions; leaves `a` does not touch keep their prices;
3. re-runs the leaf closure on every leaf, since the center changed;
4. has `g(s') = g(s) + cost(a)`.

## Goal test and search

A decoupled state is a goal state iff the center goal holds and every leaf has
a reached rank satisfying its goal. Its *solution cost* is `g` plus, per leaf,
the cheapest price among goal ranks.

Milestone 1 search is uniform-cost search over decoupled states ordered by
`g`. Because the solution cost exceeds `g` by the leaf goal prices, the first
goal state found is not necessarily optimal. The search records the best
solution at generation time and stops once the popped `g` is at least the
best solution cost (leaf prices are non-negative).

## Dominance pruning

Prices can grow without bound along center cycles: a center loop that forces
a leaf to alternate between two states raises both prices on every round.
Exact duplicate detection then never terminates on unsolvable tasks.

Milestone 1 prunes a newly generated state `t` if a stored state `s` (open or
closed) has the same center, `g(s) <= g(t)`, and `prices_s[L][r] <=
prices_t[L][r]` for all leaves and ranks (unreached as `∞`). Every completion
of `t` is then also a completion of `s` at no higher cost, so this is safe for
optimality. Termination follows from Dickson's lemma: the comparison is a
well-quasi-order on `(g, prices)` per center state, so no infinite sequence of
pairwise non-dominated states exists.

The store is a list per center state, scanned linearly, and dominated nodes
already in the open list are not removed. Milestone 2 replaces it with the
stronger state-dominance criteria of Torralba et al. (2016); which of their
criteria to implement (and how leaf-state simulation fits in) is still to be
read from the paper.

## Solution reconstruction

The search keeps parent pointers between decoupled states (global action
plus predecessor). Leaf paths are recomputed only for the solution:

1. per leaf, pick the cheapest goal rank in the final decoupled state;
2. walk the stages backwards: re-run the closure of that stage with parent
   pointers (updated only on strict improvement, so zero-cost operators cannot
   create cycles), and trace from the target rank back to a seed;
3. step across the global action: the predecessor rank must satisfy the
   action's leaf precondition, map to the seed under its leaf effects, and
   carry exactly the seed's price;
4. emit the plan as: stage 0 leaf actions of every leaf, global action 1,
   stage 1 leaf actions, and so on. Leaf actions of different leaves within a
   stage commute, since each reads only the fixed center and its own leaf.

The plan cost equals the reported solution cost; tests check that with
`Task::validate_plan`.

## Heuristic interface

Milestone 3. A decoupled state stands for many explicit states, so the
heuristic must be admissible for all of them at once. The planned trait:

```rust
pub trait DecoupledHeuristic {
    /// Lower bound on the remaining cost of the best completion of `state`,
    /// counting global actions and leaf prices still to pay; `None` if
    /// provably unsolvable.
    fn estimate(&self, view: &DecoupledView<'_>) -> Option<u64>;
}
```

Candidates: `h^max` and LM-cut adapted to decoupled states (Gnad and
Hoffmann 2015 evaluate `h^max` in fork-decoupled search; how they fold leaf
prices in is not yet checked), and abstraction heuristics from
`merge-and-shrink` behind an optional feature. Search becomes A* on
`g + h`, with the same "stop when popped f >= best solution" rule.

## Unsupported input

Milestone 1 rejects tasks with axioms or derived variables
(`Error::Axioms`) and operators with conditional effects
(`Error::ConditionalEffects`). Conditional effects on a leaf are
manageable (evaluate per rank); conditions spanning leaves are not, and would
have to turn the operator into a global action with per-leaf case splits.
Invalid tasks are rejected via `Task::validate`.

## Testing strategy

- **Cross-check against explicit-state Dijkstra.** On random tasks from
  `sas_plus::random::random_task` (no axioms, no conditional effects), compare
  solvability and optimal cost between a plain Dijkstra over `sas_plus::State`
  and decoupled search. Random tasks mostly have one causal-graph component,
  so the fork factoring alone would test nothing; the tests also run every
  seed under several explicit partitions (single-variable leaves, pairs,
  center-empty) and assert that enough solvable, nonzero-cost cases with at
  least two leaves were seen. A variant zeroes some operator costs, to
  exercise ties in leaf closure and reconstruction.
- **Fork shape.** A hand-built truck-and-packages task checks that the fork
  factoring finds one center and two leaves.
- **Plan validity.** Every returned plan passes `Task::validate_plan` with the
  reported cost.
- **Termination on unsolvable tasks.** A hand-built task whose center loop
  forces a leaf to alternate, with an unreachable goal.
- **Fixtures.** The hand-encoded gripper tasks from `sas-plus` (copied to
  `tests/fixtures`, same license), with known optimal costs 7 and 5.
- Seeds are looped deterministically; no proptest dependency.

## Milestones

1. **Done in this draft.** Fork and explicit factorings, decoupled state with
   dense prices, leaf closure, global successors, uniform-cost search with
   simple dominance pruning, plan reconstruction, cross-checks.
2. Sparse price maps; leaf-state dominance and frontier pruning
   (Torralba et al. 2016); expansion statistics.
3. Incident-arc / SCC star factorings with a scoring function; A* with a
   heuristic trait; `h^max` over the center.
4. Satisficing mode (reachability only, no prices; GBFS); optional
   `merge-and-shrink` heuristics behind a feature.
5. Conditional effects confined to one leaf.

## Cross-crate needs

- `sas-plus`: no SCC computation on `CausalGraph`; this crate computes sink
  components itself. A `CausalGraph::strongly_connected_components` would
  serve this crate and `merge-and-shrink`.
- `sas-plus`: `State` does not implement `Ord`, so `no_std` maps keyed by
  states need a `Vec<Value>` key. An `Ord` derive would help.
- `heuristic-search`: once it exists, the test-only Dijkstra here should be
  replaced by its implementation, and decoupled search could reuse its open
  lists.
- `search-core`: a shared `Plan { ops, cost }` type would avoid one more
  local definition.

## References

- Daniel Gnad and Jörg Hoffmann. *Beating LM-Cut with h^max (Sometimes):
  Fork-Decoupled State Space Search.* ICAPS 2015, pp. 88–96.
- Daniel Gnad and Jörg Hoffmann. *Star-Topology Decoupled State Space
  Search.* Artificial Intelligence 257 (2018), pp. 24–60.
  doi:10.1016/j.artint.2017.12.004.
- Daniel Gnad, Valerie Poser and Jörg Hoffmann. *Beyond Forks: Finding and
  Ranking Star Factorings for Decoupled Search.* IJCAI 2017, pp. 4310–4316.
- Álvaro Torralba, Daniel Gnad, Patrick Dubbert and Jörg Hoffmann. *On
  State-Dominance Criteria in Fork-Decoupled Search.* IJCAI 2016,
  pp. 3265–3271.
