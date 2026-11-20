# decoupled-search: design

Decoupled (star-topology) search for factored SAS+ planning tasks from
[`sas-plus`](https://crates.io/crates/sas-plus). The crate partitions a
task's variables into one *center* and several *leaves*, searches over center
paths only, and keeps, per leaf, the set of leaf states reachable along the
current center path together with their cheapest cost (the *price*).

Status: milestones 1 to 5 (see [Milestones](#milestones)).

## Terminology

| Term | Meaning |
| --- | --- |
| Factoring | A partition of the state variables into a center `C` and leaves `L1..Lk`. `C` may be empty, `k` may be 0. |
| Leaf state | An assignment to the variables of one leaf, stored as a mixed-radix rank. |
| Leaf action of `L` | An operator whose effects all lie in `L` (at least one), and whose preconditions and effect conditions all lie in `C ∪ L`. |
| Global action | Every other operator: it changes the center, or touches two leaves, or reads another leaf. |
| Decoupled state | A center state plus, per leaf, a price for every leaf state (`∞` = unreached). |
| `g` | The cost of the global actions on the path to a decoupled state. |
| Price | The cost of the cheapest leaf-action sequence that reaches a leaf state, interleaved with the current global path. |

## Factoring strategies

The classification above is our own formulation. It makes every variable
partition a valid factoring: an operator that would break the star topology
(for example one that reads leaf `L1` and writes leaf `L2`) becomes a global
action, so it is searched explicitly instead of being priced. The factoring
therefore only affects performance, never soundness or optimality. Gnad,
Poser and Hoffmann (2017) define star factorings by: every action that does
not affect the center affects only one leaf and has preconditions only on
it and the center. Our leaf actions are exactly those actions (their
*leaf-only* actions); we additionally accept actions that affect the center
or several leaves, and actions that read another leaf, as global actions,
so arbitrary partitions are searchable. The strategies below produce
*strict-star* factorings (every causal-graph arc between factors touches
the center), which are star factorings.

Strategies:

1. **Fork (sink components)**, milestone 1, `Factoring::fork`. Every sink
   strongly connected component of the causal graph (no arcs leaving it) whose
   leaf state space has at most `2^16` states becomes a leaf; everything else
   is the center. This is the refined fork strategy of Gnad, Poser and
   Hoffmann (footnote 1), which maximises the number of fork leaves.
   Operators on a sink component read only the center and that
   component, so all of them are leaf actions. On a task whose causal graph is
   one component, this yields no leaves and the search degenerates to plain
   uniform-cost search over the center.
2. **Explicit**, milestone 1, `Factoring::new(task, leaves)`. The caller names
   the leaves; the center is the rest. Used by tests to exercise arbitrary
   partitions.
3. **Inverted fork**, `Factoring::inverted_fork`. The fork strategy on the
   reversed causal graph: every small source SCC (no arcs entering it)
   becomes a leaf (the paper's refined inverted fork).
4. **X-shape**, `Factoring::x_shape` (Gnad, Hoffmann and Domshlak, SoCS
   2015). The fork leaves, plus every small source SCC that has no arc
   into a fork leaf (the inverted fork run on the fork center; fork leaves
   have no outgoing arcs, so these are source SCCs of the whole graph).
5. **Incident arcs**, `Factoring::incident_arcs` (the paper's Figure 2).
   Sort the variables by decreasing number of incident causal-graph arcs
   (in plus out), move them into the center one at a time, take the
   weakly connected components of the rest (within `2^16` states) as the
   leaves, and keep the candidate with the most mobile leaves.
6. **Independent set**, `Factoring::max_leaves` (Figure 1). Seed one leaf
   per variable of a maximum independent set of the causal graph, grow the
   leaves by variables adjacent to exactly one leaf (`maximizeFlexibility`),
   drop frozen leaves, grow again, and keep the result with the most mobile
   leaves over all enumerated maximum independent sets. Branch and bound
   enumerates up to 16 sets within a budget of `2^24` neighbour lookups
   (deterministic, instead of the paper's ten-second time-out); a unit
   test checks it against brute force. Growing tracks mobility
   incrementally: without axioms, moving `v` into leaf `L` can only change
   `L`'s mobility, through operators touching `v`.

`Factoring::check_star` reports the first causal-graph arc between two
leaves (`Error::NotStar`). Every strategy except `Factoring::new` passes
it; search does not require it. Derived variables are never leaf
variables: the strategies move them to the center (see
[Conditional effects and axioms](#conditional-effects-and-axioms)).

*Scoring.* `Factoring::score` compiles the factoring with the search's own
operator classification and reports the number of leaves, *mobile* leaves
(leaves with at least one leaf action: the paper's flexibility, the share
of leaf-only actions among the actions affecting the leaf, above 0), leaf
and global actions, the leaf-action share, and the largest and total leaf
size. `Factoring::auto` scores every `Strategy`, keeps those with at least
two mobile leaves (the paper abstains below two, since decoupling pays off
by not enumerating combinations of leaf states across leaves), and picks
the best by the larger leaf-action share (an exact fraction), more mobile
leaves, the smaller largest leaf, and the earlier strategy. Factorings the
search would reject are skipped; if none is left, the result is the
factoring without leaves (explicit search). The paper ranks by one-second
sample searches instead; see the table.

*Checked against the paper* (Gnad, Poser and Hoffmann, IJCAI 2017):

| Paper | Paper rule | Our code | Status |
| --- | --- | --- | --- |
| Sec. 3.3, fn. 1 | refined fork / inverted fork: leaves are the sink / source causal-graph SCCs | `fork`, `inverted_fork`, components over `2^16` states join the center | matches (leaf bound: see below) |
| Sec. 3.3 (and Gnad, Hoffmann and Domshlak 2015) | X-shape: fork strategy, then inverted fork on the fork center; inverted-fork leaves with an arc into a fork leaf go back to the center | `x_shape`, on the refined fork and inverted fork | matches, with the refined (maximum-leaves) fork of footnote 1 instead of the 2015 greedy horizontal line |
| Sec. 4 | leaf-size bound `2^32` states | `MAX_LEAF_STATES = 2^16` | deliberate deviation: leaf states are `u16` ranks |
| Sec. 4.1 | mobile: flexibility (leaf-only actions over all actions affecting the leaf) above 0; frozen otherwise | at least one leaf action (`score`, and a cheap count the strategies use, asserted equal) | matches |
| Fig. 2 | incident arcs: static order on the full causal graph, one candidate per center size 1 to the number of variables, max mobile leaves | same; ties: smaller variable, then smaller center | bug fixed: the order was recomputed per step from neighbours outside the center (distinct neighbours, not arcs), and the first candidate with two leaves (mobile or not) was taken; regression test `incident_arcs_keeps_the_most_mobile_leaves` |
| Fig. 1 | MIS seed, `maximizeFlexibility`, drop frozen, `maximizeFlexibility` | `max_leaves`, as listed above | bug fixed: `max_leaves` was the incident-arcs walk with the most (not mobile) leaves, with no independent set and no growing; regression test `max_leaves_grows_the_independent_set` |
| Fig. 1 | per round an argmax over the round's leaf sets (pseudocode: their size; prose: mobile leaves); return the round with the most mobile leaves | per round the most mobile leaves (the prose; the number of leaves is constant while growing), smallest variable on ties; between rounds the latest on ties | deliberate reading, ties ours |
| Sec. 4.1 | all maximum independent sets found within 10 s | up to 16 within a deterministic work budget | deliberate deviation: `no_std`, reproducible |
| Sec. 4 | abstain below two mobile leaves | `incident_arcs` and `max_leaves` return their best candidate anyway; `auto` abstains (no leaves) | deliberate: the strategy functions return a `Factoring`, not an `Option`; `auto` used to accept a single mobile leaf |
| Sec. 5 | pick a factoring by 1 s sample searches: expansions (optimal), heuristic improvement (satisficing) | static score: leaf-action share, mobile leaves, largest leaf | deliberate deviation: no clock in `no_std`, and `auto` does not know the search configuration; open question |
| — | no axioms | the inputs of a derived variable count as adjacent in both strategies, so none is split over two leaves (`Error::AxiomAcrossLeaves`) | extension; regression test `strategies_keep_axiom_inputs_in_one_factor` |
| Sec. 2 | causal graph arcs `pre ∪ eff -> eff` | `sas_plus::CausalGraph`, plus effect conditions and axioms | matches (the extra arcs only keep conditional effects and axioms inside one factor) |

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
ranks and `u64` prices, ten bytes per reached state. When every reached
state has price 0 (always, with reachability-only leaves) the price vector
is dropped, two bytes per reached state. Leaf closure,
transfers and reconstruction work on a dense vector that successor
generation builds from the stored form and compacts again; that costs
`O(|S_L|)` time per leaf per successor, but no longer `O(|S_L|)` memory per
stored state.

## Successor generation

A global action `a` is applicable in decoupled state `s` iff its center
preconditions hold in `s.center` and, for each leaf where it has a
precondition, at least one reached leaf state satisfies it. The successor:

1. for each leaf `a` reads or writes: keeps the reached ranks satisfying
   `pre_L(a)`, maps each through `eff_L(a)` (conditional effects evaluated
   on that rank and the predecessor center), and keeps the minimum price on
   collisions; leaves `a` does not touch keep their prices;
2. applies `a`'s center effects (conditions read from the predecessor
   center), then re-evaluates the derived center variables;
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

`Config::anytime_astar` is Gnad and Hoffmann's Anytime Fork-Root A\*
(ICAPS 2015, Figure 4), generalised to star factorings:

| Paper | Paper rule | Our code | Status |
| --- | --- | --- | --- |
| Fig. 4 | A\* ordered by a root heuristic `h^R` (remaining center cost) | key `g + root`, where `root` sees every reached fact at price 0 | matches |
| Def. 6 | leaves-for-free root estimate: `h` on the task with leaf-action costs 0 | `Factoring::with_free_leaf_actions`, for example under `HMax` | matches |
| Fig. 4 | discard generated `N` with `g + h^F >= U` | same, counted in `Statistics::bound_pruned` | matches |
| Fig. 4 | goals checked on expansion; `U` updated; stop at an expanded goal with goal price `MINGprice`, else exhaust the open list | goals recorded on generation; stop once popped key + `MINGprice >= U` | deliberate: subsumes the paper's early termination (its goal has key `g`, so `U <= key + MINGprice`) and is sound by the A\* covering argument |
| Fig. 4 | `MINGprice`: sum over leaves of the optimal cost of the projection onto the leaf | per leaf the cheapest leaf-action cost from the initial to a goal leaf state, global parts free, center ignored, reduced transitions skipped; 0 for leaves whose transitions read the center | generalisation (global actions move leaves at no price), a lower bound on every goal price |
| Thm. 5 | optimal for root-admissible `h^R` and fork-admissible `h^F` | same; checked against Dijkstra with a blind root and `h^max` on the free-leaf task in every optimal configuration | matches |
| Fig. 4 | Fork-Root A\* (FRA\*, not complete) | not implemented | not needed: anytime A\* is complete |

## Dominance pruning

Prices can grow without bound along center cycles: a center loop that forces
a leaf to alternate between two states raises both prices on every round.
Exact duplicate detection then never terminates on unsolvable tasks, so
uniform-cost search, A* and GBFS prune a generated state `n` if a stored
state `t` (open or closed) with the same center *dominates* it: every
completion of `n` maps to a completion of `t` with the same remaining
global actions at no higher total cost. Dominated nodes already in the open
list are not removed, and the store is a list per center state.

`Config::pruning` selects one of the criteria of Torralba, Gnad, Dubbert and
Hoffmann (IJCAI 2016), checked against the paper and its extended
technical report (see the table below) and generalised to this crate's
global actions. Each gives a bound `D_t(x)` for every leaf state `x`
reached in `n`: what `t` must pay to do whatever `n` can do from `x`.

| `Pruning` | Bound and compared states | Paper |
| --- | --- | --- |
| `Basic` | `price_t(x)`, all reached `x` | Def. 1 (Gnad and Hoffmann) |
| `Frontier` | `price_t(x)`, only on the frontier of `n` | Defs. 2, 3 |
| `EffectivePrices` | least `p` with `p = price_t` on anchored states, `p(x) = min(price_t(x), max_{x -a-> y} p(y) - cost(a))` elsewhere | Def. 4 |
| `Simulation` | `min_{x ⪯ y} price_t(y)` for the leaf simulation `⪯` | Defs. 5, 6 |
| `EffectiveSimulation` | effective prices over simulation bounds | Def. 7 |
| `Adaptive` (default) | `EffectivePrices`, then `EffectiveSimulation` once the search has spent the simulation's estimated cost | ours |

*Generalisation.* The paper assumes fork factorings with a single goal leaf
state, where only leaf actions move a leaf. Here global actions may read
and write leaves, and leaf goals are partial. A leaf state is *anchored* if
it satisfies the leaf goal or the leaf preconditions of some global
action's part on the leaf (all states are, if a part has no leaf
precondition). The frontier contains the anchored states; effective prices
stop their recursion there; the simulation must match every global part
with the same action (preconditions and successors). An unanchored state
can only be left by leaf actions, which is what the paper's proofs use, so
they carry over; the dominance module's rustdoc writes them out for this
setting. Leaves whose transitions depend on the center (center-conditioned
effects, derived slots) or that exceed `Config::max_analysis_states`
(default 256) use the basic bound.

*Leaf simulation* (`search/dominance/simulation.rs`) is the coarsest
relation with: goal leaf states are simulated only by goal leaf states;
global parts are matched by the same part; a leaf action `x -a-> x'` is
matched by staying (`x' ⪯ y`) or by a leaf action `b` with `cost(b) <=
cost(a)` and center preconditions contained in `a`'s, `y -b-> y'`, `x' ⪯
y'`. It is computed once per leaf by worklist refinement.

*Folding `g` in (the open question of the earlier design).* Leaf prices
exclude `g`. The classic rule, `g(t) <= g(n)` and every bound at most the
price, is used with `fold_g = false`. Comparing `g + price` per leaf state
would be unsound with two or more leaves: `g` is paid once but credited
once per leaf (`t`: `g = 0`, prices 10 and 10; `n`: `g = 10`, prices 0 and
0; per leaf both sum to 10, but completing both leaves costs 20 from `t`
and 10 from `n`). With `fold_g = true` (default) the budget is shared:
with `δ_L = max_x (D_t(x) - price_n(x))`, `t` dominates `n` iff
`g(t) + Σ_L δ_L <= g(n)`. Every completion of `n` through leaf states
`x_L` then has one of `t` cheaper by at least `g(n) - g(t) - Σ_L δ_L >= 0`.
This subsumes the classic rule and preserves optimality.

*Optimality of A\*.* A stored node covers cost `C` via global actions `π`
if `π` leads it to a goal with total cost at most `C`. While `C*` is not
recorded, a covering node with the shortest `π` is open: were it closed,
its successor along `π[0]` was stored (shorter cover), pruned by a node
covering via the same shorter suffix (dominance keeps the remaining global
actions, so no `d*` condition as in the paper is needed), or a dead end of
an admissible heuristic (impossible). So the search does not stop with a
popped key `>= best` before `best = C*`.

*Greedy search* only needs reachability: `t` dominates `n` if every
compared bound is finite, whatever the costs.

*Termination* follows from Dickson's lemma for the basic criterion, which
every other one subsumes: the comparison is a well-quasi-order on
`(g, prices)` per center state, so no infinite sequence of pairwise
non-dominated states exists.

*Adaptive default.* Simulation costs `O(size^2 * actions)` per leaf once
per search, which dominated small searches (gripper: 1.24 ms against
24 µs for `Basic`). `Pruning::Adaptive` follows the ski-rental rule: it
uses effective prices, counts one work unit per leaf state of every
generated decoupled state, and computes the simulations and switches to
`EffectiveSimulation` once the work reaches `Σ size^2 * (1 + leaf actions)`
over the analysed leaves. Mixing sound criteria is sound; cached bounds
are recomputed after the switch. Measured on gripper (fork, uniform-cost,
release, criterion `--quick`): `Basic` 22 µs, `Adaptive` 40–57 µs,
`EffectiveSimulation` 700–750 µs.

*Leaf reduction* (`Config::reduce_leaves`, default off). Section 6 of
the extended technical report calls a leaf transition `x -a-> x'`
superfluous if `x' ⪯ x`, or if another transition `x -b-> y'` has
`x' ⪯ y'`, center preconditions contained in `a`'s and `cost(b) <=
cost(a)`; it then removes unreachable leaf states and leaf actions without
transitions. It gives no removal order and no proof. Read as one
simultaneous step the rule is unsound: two transitions that replace each
other (two copies of one action) would both go. We remove `a` under the
second condition only if `a` does not replace `b` in turn or `b` has the
smaller index, which is what one-at-a-time removal, each step justified by
a transition still present, removes (up to which of two mutually replacing
transitions stays). Leaf closure skips removed transitions, so leaf states
only reachable through them are never reached. `Config::reduced_task`
returns the task without the leaf actions that keep no transition from a
leaf state the reduced leaf can still reach (under any center), for
building heuristics: the report's gain from removed actions (LM-cut in
Logistics98). It is sound only together with `reduce_leaves` in the
search: every reached leaf state is then reduced-reachable, and every
decoupled state has a cheapest completion over kept transitions from such
states, so a heuristic admissible on the reduced task is admissible for
the search. A random cross-check runs A\* with `h^max` of the reduced task
against Dijkstra. Soundness (our proof, on
`Config::reduce_leaves`): "replaces" is a preorder, the index tie-break
makes the removal order a strict partial order, so the first removed step
of any compliant leaf path can be replaced by a kept step or by staying,
and the rest of the path is mirrored by the simulation; repeating
terminates, so the reduced system loses no completion and no cost. On the
Logistics example it removes the five unloads at non-goal locations and
the load at the goal (those states become unreachable) and stores fewer
decoupled states. Reconstruction runs the same reduced closure, so plans
stay consistent.

*Checked against the papers.* Torralba, Gnad, Dubbert and Hoffmann, IJCAI
2016 and its extended technical report (the definitions are identical in
both; the proofs and the leaf reduction are in the report only):

| Paper | Paper rule | Our code | Status |
| --- | --- | --- | --- |
| Def. 1 | `s ⪯_B t` iff same center and `prices(s) >= prices(t)` everywhere | `Pruning::Basic`, all reached states of `n` | matches |
| Def. 2 | frontier: the goal leaf state, plus `x` with a leaf transition `x -a-> y`, `price(x) + cost(a) < price(y)` (transitions ignore the center) | anchored states (goal *or* a global part's leaf precondition) plus the same improvement test over leaf actions under any center | deliberate deviation: global actions may read and write leaves, so a leaf state can also matter by enabling one; the report's proof (last state on the path whose price equals the path cost) then ends in an anchored state |
| Def. 3 | compare prices on the frontier of `s` | `Pruning::Frontier` | matches |
| Def. 4 | `Eprices(t)`: least `p` with `p = price` at the goal, `min(price, max_{x -a-> y} p(y) - cost(a))` elsewhere | `effective_prices`, anchored states keep the base bound; max-heap Dijkstra backwards | deliberate deviation (anchored for goal), as Def. 2 |
| Def. 5 | leaf simulation: goal only simulated by goal; each `x -a-> x'` matched by staying or by `y -b-> y'`, `pre_C(b) ⊆ pre_C(a)`, `cost(b) <= cost(a)`; coarsest one | `simulation::simulators`, coarsest by worklist refinement; partial goals (goal states only simulated by goal states) and every global part matched by itself | deliberate deviation (global parts, partial goals), proof in the module rustdoc |
| Def. 6 | `min_{x ⪯ y} price(t)(y)` | `simulation_bounds` | matches |
| Def. 7 | `ESprices`: as Def. 4 with the simulation minimum off the goal, the plain price at the goal | the simulation minimum on anchored states too | deliberate deviation, sound by Theorem 5's argument (Lemma 1 holds from every leaf state); it can only prune more |
| Sec. 3 | prune `s` if a seen `t` has `s ⪯ t` and `g(s) >= g(t)`; dominance also requires `d*(s) >= d*(t)` for 0-cost actions | `fold_g = false`: `g(t) <= g(n)` and every bound at most the price; `fold_g = true` (default): one budget `g(t) + Σ_L δ_L <= g(n)` | deliberate extension (subsumes the paper's rule); no `d*` condition, because every criterion maps a completion to one with the same remaining global actions, see *Optimality of A\** |
| Sec. 3 | fork factorings, one goal leaf state, center actions do not touch leaves | star factorings with global actions on leaves, partial leaf goals; leaves with center-dependent transitions or more than `max_analysis_states` states use Def. 1 | deliberate generalisation, proofs on `Pruning` |
| Sec. 6 (report) | superfluous transitions, then unreachable leaf states, then actions without transitions | rule 2 with the removal order above; states stay but are never reached; `Config::reduced_task` drops the actions for heuristics | bug in the literal rule avoided; ours strengthened to match one-at-a-time removal (before, equal-cost and equal-precondition replacements were kept whenever the replacing action had the larger index; regression test `one_directional_replacement_removes_the_dominated_transition`) |
| Sec. 8 | DA\* and anytime ADA\* | `astar` (DA\*, stopping once the popped key reaches the best solution cost recorded at generation) and `anytime_astar` (ADA\*, see *Goal test and search*) | matches |

## Reachability-only leaves

`Leaves::Reachability` closes leaves with every leaf action at cost 0, so
a decoupled state records only which leaf states are reached. States are
smaller and dominance is plain set inclusion (modulo simulation). This
loses costs, so it is refused for cost-optimal search
(`Error::ReachabilityNotOptimal`); GBFS uses it, and the plan's cost is
summed from its operators. Solvability is unchanged: tests compare it with
Dijkstra under every criterion.

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

The plan cost equals the reported solution cost (a debug assertion checks
it). `Plan::validate` checks plans with explicit-state sas-plus semantics,
independent of the decoupled machinery, and reports the first problem as a
`PlanError`.

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
state for leaf facts, and records the leaf of every variable
(`FactPrices::leaf_of`). The contract: a member state costs at least the sum
over factors (the center counts as one) of the largest price of its facts
in that factor. `FactPrices::new` and `FactPrices::from_state` build them
for heuristic tests.

- `HMax` seeds `h^max` over all operators of the original task with these
  prices. It is admissible: a completion through a member state pays at
  least the largest price of a fact it uses, plus at least `h^max` of that
  state.
- `HFf` extracts a relaxed plan from `h^add` best supporters and adds the
  largest seed price it starts from; it is not admissible and meant for
  GBFS.
- `MasHeuristic` (feature `merge-and-shrink`) evaluates a merge-and-shrink
  abstraction's merge tree bottom-up on fact prices: per tree node and
  abstract state the cheapest price bound, summing children that touch
  disjoint factors and taking the maximum when they share one; the estimate
  is the cheapest bound plus abstract goal distance at the root. The proof
  is in the module rustdoc. `MergeAndShrinkHeuristic` keeps its tree
  private, so the adapter builds the abstraction from the crate's public
  `Factor`, `MergeStrategy` and `ShrinkStrategy` (label reduction is not
  public and is skipped).

Gnad and Hoffmann (2015, Definition 5) fold leaf prices in by a task
`Π^L$` whose leaf variables start unknown and with one new action per
reached leaf state, setting that state at its price (once per leaf). In
the delete relaxation the once-per-leaf flag never blocks, so `h^max` on
`Π^L$` gives each leaf fact the cheapest price of a reached leaf state
containing it, which is exactly our seed: `HMax` matches their
`h^max_{L$}` (checked against the paper). `HFf` uses the same seeds and
adds the largest seed price its relaxed plan starts from, our own choice
rather than `h^FF` on `Π^L$`; it is not admissible either way. Their root estimate
(Definition 6) is `Factoring::with_free_leaf_actions`. LM-cut remains
open.

## Conditional effects and axioms

*Conditional effects.* An effect fires when its conditions hold in the
state the operator is applied in; effects apply in operator order and a
later firing effect on the same variable wins, as in `sas_plus::Task::apply`.
Decoupled search evaluates them per leaf state where it can:

- an effect on leaf `L` may be conditioned on the center and on `L`. It is
  evaluated per rank, on the center the operator is applied in (fixed
  during leaf closure, the predecessor center for a global action);
- an effect on the center may be conditioned only on the center. A
  condition on a leaf would make the successor center depend on the leaf
  state (`Error::CenterEffectReadsLeaf`);
- an effect on leaf `L` conditioned on another leaf would couple the two
  leaves (`Error::EffectConditionAcrossLeaves`).

An operator is a leaf action of `L` if its preconditions, effects and
effect conditions all lie in `C ∪ L` and it sets something in `L` but
nothing in `C`.

*Axioms.* Derived variables are never leaf variables (`Factoring::new`
rejects them with `Error::DerivedInLeaf`, the strategies keep them in the
center). The compiler follows each derived variable's rules transitively to
the state variables they read:

- only center variables: the derived variable is a center slot,
  re-evaluated after every center change;
- the center and one leaf `L`: it is *attached* to `L` as an extra slot
  whose value is evaluated per rank under the current center (a `View`
  evaluates all attached slots once per center). Preconditions, goals and
  effect conditions on it are facts on `L`;
- two leaves: `Error::AxiomAcrossLeaves`.

Evaluation mirrors `sas_plus::Task::evaluate_axioms` on a scratch
assignment, restricted to the derived variables of one factor, which is
closed under dependencies; a unit test compares it with sas-plus on random
states. Leaves with attached derived variables cost `O(|S_L| * rules)` per
closure and transfer for the evaluation.

*Heuristics.* `h^max` and `h^FF` split operators into one relaxed operator
for the unconditional effects and one per conditional effect (preconditions
plus conditions), add every axiom as a 0-cost relaxed operator, and seed the
default value of every derived variable at cost 0: no relaxed operator
produces a default value, and without the seed a goal on it would make
`h^max` declare a solvable state a dead end.

Invalid tasks are rejected via `Task::validate`.

## Testing strategy

- **Cross-check against explicit-state Dijkstra.** On random tasks from
  `sas_plus::random::random_task`, compare solvability and optimal cost
  between a plain Dijkstra over `sas_plus::State` and decoupled search.
  Every check runs uniform-cost search and A* with `h^max` (and the
  merge-and-shrink adapter, exact and size-limited, with the feature)
  under all ten pruning configurations (five criteria, with and without
  `fold_g`). Random tasks mostly have one causal-graph component, so the
  tests also run every seed under several explicit partitions and assert
  that enough solvable, nonzero-cost cases with at least two leaves were
  seen. Variants zero some operator costs, and add conditional effects and
  derived variables: unmodified, every factoring is either rejected with a
  precise error or matches; with the inexpressible conditions dropped per
  factoring, every factoring must be accepted and match, with a floor on
  runs that attach derived variables to leaves.
- **Admissibility on member states.** A unit test enumerates the member
  states of the first decoupled states of random tasks and checks `h^max`
  and the merge-and-shrink adapter against the cheapest member price plus
  `h*` from `Task::perfect_heuristic`.
- **Dominance non-vacuity.** The Logistics example of Torralba et al.
  shows each stronger criterion avoiding the basic blow-up; aggregated
  random runs show each setting pruning what the weaker one keeps; a unit
  test pins the two-leaf counterexample to per-leaf `g` folding; the leaf
  simulation is hand-checked on a Logistics leaf.
- **Fork shape**, **star tasks** (all strategies and `auto`), **scoring**.
- **Plan validity.** Every returned plan passes `Plan::validate`, which is
  itself checked against `Task::validate_plan`.
- **Greedy search** with `h^FF`, `h^max`, and reachability-only leaves
  under every criterion: solvability as Dijkstra, never below the optimum.
- **Termination on unsolvable tasks** and **fixtures** (gripper, optimal
  costs 7 and 5).
- **Axiom evaluation** against `Task::evaluate_axioms` on random states.
- Seeds are looped deterministically; no proptest dependency. Tests build
  with `opt-level = 1`.

## Milestones

1. **Done.** Fork and explicit factorings, leaf closure, global
   successors, uniform-cost search with simple dominance pruning, plan
   reconstruction, cross-checks.
2. **Done.** Sparse stored prices; frontier, effective-price and
   simulation dominance with a shared `g` budget; search statistics.
3. **Done.** Inverted-fork and greedy star factorings with a star check;
   A* with a heuristic trait; `h^max` and `h^FF` seeded with fact prices;
   factoring scores and `Factoring::auto`.
4. **Done.** GBFS; reachability-only leaves; the `merge-and-shrink`
   heuristic adapter.
5. **Done.** Conditional effects and axioms where decoupling allows;
   `Plan::validate`; a public `FactPrices` constructor.

Done since: simulation-based leaf reduction and the adaptive default;
the factoring strategies, the leaf reduction and the dominance criteria
checked against the papers (tables above), with the incident-arcs and
independent-set strategies of Gnad, Poser and Hoffmann (2017).
Open: LM-cut, an index for the dominance store, decoupled symbolic
search, decoupled partial-order reduction (strong stubborn sets), and
multiple centers.

## Cross-crate needs

- `sas-plus`: no SCC computation on `CausalGraph`; this crate computes sink
  components itself. A `CausalGraph::strongly_connected_components` would
  serve this crate and `merge-and-shrink`.
- `sas-plus`: `State` does not implement `Ord`, so `no_std` maps keyed by
  states need a `Vec<Value>` key. An `Ord` derive would help.
- `heuristic-search`: once it exists, the test-only Dijkstra here should be
  replaced by its implementation, and decoupled search could reuse its open
  lists.
- `search-core`: a shared `Plan { ops, cost }` type (with validation) would
  avoid one more local definition.
- `merge-and-shrink`: an accessor for `MergeAndShrinkHeuristic`'s final
  `Factor` (or its merge tree) would let `MasHeuristic` wrap the crate's own
  `build`, with label reduction, instead of re-running the merge loop from
  the public parts (including a copy of its private `shrink_sizes`).
- `planning-heuristics`: `HMax` and `HFf` here run their own relaxed
  exploration. If that crate offers `h^max`/`h^add` seeded with per-fact
  initial costs, these could delegate to it.

## References

- Daniel Gnad and Jörg Hoffmann. *Beating LM-Cut with h^max (Sometimes):
  Fork-Decoupled State Space Search.* ICAPS 2015, pp. 88–96 (Anytime
  Fork-Root A\*, Figure 4).
- Daniel Gnad and Jörg Hoffmann. *Star-Topology Decoupled State Space
  Search.* Artificial Intelligence 257 (2018), pp. 24–60.
  doi:10.1016/j.artint.2017.12.004.
- Daniel Gnad, Jörg Hoffmann and Carmel Domshlak. *From Fork Decoupling
  to Star-Topology Decoupling.* SoCS 2015.
- Daniel Gnad, Valerie Poser and Jörg Hoffmann. *Beyond Forks: Finding and
  Ranking Star Factorings for Decoupled Search.* IJCAI 2017, pp. 4310–4316.
- Álvaro Torralba, Daniel Gnad, Patrick Dubbert and Jörg Hoffmann. *On
  State-Dominance Criteria in Fork-Decoupled Search.* IJCAI 2016,
  pp. 3265–3271. Extended technical report with the proofs and the leaf
  reduction rule:
  <https://fai.cs.uni-saarland.de/hoffmann/papers/ijcai16b-tr.pdf>.
