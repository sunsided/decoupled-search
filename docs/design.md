# decoupled-search: design

Decoupled (star-topology) search for factored SAS+ planning tasks from
[`sas-plus`](https://crates.io/crates/sas-plus). The crate partitions a
task's variables into one *center* and several *leaves*, searches over center
paths only, and keeps, per leaf, the set of leaf states reachable along the
current center path together with their cheapest cost (the *price*).

Status: milestone 1 plus parts of milestones 2 to 4 (see
[Milestones](#milestones)).

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
3. **Inverted fork**, `Factoring::inverted_fork`. The fork strategy on the
   reversed causal graph: every small source SCC (no arcs entering it)
   becomes a leaf.
4. **Greedy strict-star factorings**, `Factoring::incident_arcs` and
   `Factoring::max_leaves`. Starting from an empty center, move the variable
   with the most causal-graph neighbours outside the center into the center,
   one per step, until all variables are in it. At each step the candidate
   leaves are the weakly connected components of the remaining variables
   (components over `2^16` states join the center), so no arc joins two
   leaves. `incident_arcs` takes the first step with at least two leaves
   (else falls back to `max_leaves`), `max_leaves` the step with the most
   leaves, the smaller center on ties. These are LP-free and only in the
   spirit of Gnad, Poser and Hoffmann (2017), whose exact strategies (and
   their scoring by leaf actions and leaf sizes) are not checked against
   the paper.

`Factoring::check_star` reports the first causal-graph arc between two
leaves (`Error::NotStar`). Every strategy except `Factoring::new` passes
it; search does not require it.

## Decoupled state and pricing

```text
DecoupledState {
    center: Vec<Value>,      // values of the center variables, in Factoring::center order
    prices: Vec<Prices>,     // per leaf: reached ranks (u16, ascending) and their u64 prices
}
```

Leaf states are ranked in mixed radix over the leaf's variables, so a leaf
with variables of domain sizes `d0, d1, ...` has `d0 * d1 * ...` ranks.
Reading or setting one variable of a rank is a division, a modulus and an
add; nothing allocates per leaf state.

*Leaf closure.* Given a center state and seed prices for a leaf, Dijkstra over
the leaf's ranks with the leaf actions whose center preconditions hold. The
result is the new price vector. Seeds are all finite-priced ranks.

Stored states keep only the reached leaf states: parallel vectors of `u16`
ranks and `u64` prices, ten bytes per reached state. Leaf closure,
transfers and reconstruction work on a dense vector that successor
generation builds from the stored form and compacts again; that costs
`O(|S_L|)` time per leaf per successor, but no longer `O(|S_L|)` memory per
stored state.

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

`search` is uniform-cost search over decoupled states ordered by `g`,
`astar` orders by `g + h`. Because the solution cost exceeds `g` by the
leaf goal prices, the first goal state found is not necessarily optimal.
Both record the best solution at generation time and stop once the popped
key is at least the best solution cost (leaf prices are non-negative and
`h` is a lower bound). `gbfs` orders by `h` alone and returns the first
solution generated.

## Dominance pruning

Prices can grow without bound along center cycles: a center loop that forces
a leaf to alternate between two states raises both prices on every round.
Exact duplicate detection then never terminates on unsolvable tasks.

Uniform-cost search and A* prune a newly generated state `t` if a stored
state `s` (open or closed) has the same center, `g(s) <= g(t)`, and
`prices_s[L][r] <= prices_t[L][r]` for all leaves and ranks (unreached as
`∞`). Exact duplicates are the special case of equality. Leaf prices here
exclude `g`, so the `g` condition is needed for optimality. GBFS drops it:
`s` still reaches every goal `t` reaches, which preserves solvability. Every completion
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

The plan cost equals the reported solution cost; tests check that with a
test-local validator that applies operators fact by fact.

## Heuristic interface

A decoupled state stands for many explicit states, so the heuristic must
be admissible for all of them at once:

```rust
pub trait DecoupledHeuristic {
    /// Lower bound on the remaining cost beyond g (leaf prices plus further
    /// operators); `None` if provably unsolvable.
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64>;
}
```

`FactPrices` gives, per fact, the cheapest price of a member state
containing it: 0 for the center facts that hold, the cheapest reached leaf
state for leaf facts. `HMax` seeds `h^max` over all operators of the
original task with these prices. It is admissible: a completion through a
member state pays at least the largest price of a fact it uses, plus at
least `h^max` of that state. `HFf` extracts a relaxed plan from `h^add`
best supporters and adds the largest seed price it starts from; it is not
admissible and meant for GBFS. How Gnad and Hoffmann (2015) fold leaf
prices into `h^max` is not checked against the paper. LM-cut and
`merge-and-shrink` heuristics remain open.

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
- **Plan validity.** Every returned plan passes a test-local validator that
  applies operators fact by fact, with the reported cost.
- **Star tasks.** Random tasks that are stars by construction (one or two
  center variables, two or three leaves, global actions reading and writing
  at most one leaf, a third of the costs zero). All strategies must pass
  `check_star`; uniform-cost search and A* with `h^max` must match Dijkstra
  under every strategy; GBFS with `h^FF` and `h^max` must agree on
  solvability and never beat the optimum.
- **Termination on unsolvable tasks.** A hand-built task whose center loop
  forces a leaf to alternate, with an unreachable goal.
- **Fixtures.** The hand-encoded gripper tasks from `sas-plus` (copied to
  `tests/fixtures`, same license), with known optimal costs 7 and 5.
- Seeds are looped deterministically; no proptest dependency.

## Milestones

1. **Done.** Fork and explicit factorings, leaf closure, global
   successors, uniform-cost search with simple dominance pruning, plan
   reconstruction, cross-checks.
2. **Partly done:** sparse stored prices. Open: the stronger dominance and
   frontier pruning of Torralba et al. (2016); expansion statistics.
3. **Done:** inverted-fork and greedy star factorings with a star check;
   A* with a heuristic trait; `h^max` and `h^FF` seeded with fact prices.
   Open: scoring factorings by leaf-action share and leaf sizes.
4. **Partly done:** GBFS. Open: reachability-only leaves without prices;
   optional `merge-and-shrink` heuristics behind a feature.
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
- `planning-heuristics`: `HMax` and `HFf` here run their own relaxed
  exploration. If that crate offers `h^max`/`h^add` seeded with per-fact
  initial costs, these could delegate to it.

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
