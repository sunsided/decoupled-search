//! A merge-and-shrink heuristic on decoupled states, from the
//! [`merge_and_shrink`] crate (feature `merge-and-shrink`). See
//! [`MasHeuristic`] for the evaluation and its admissibility proof.

use alloc::vec;
use alloc::vec::Vec;

use merge_and_shrink::{Factor, MergeAndShrink, MergeStrategy, MergeTree, ShrinkStrategy};
use sas_plus::{Fact, Task, VarId};

use super::{DecoupledHeuristic, FactPrices};

/// A price bound that no member state attains.
const INFINITE: u64 = u64::MAX;

/// A node of the merge tree, flattened in post-order.
#[derive(Clone, Debug)]
enum Node {
    /// An atomic factor over one task variable.
    Var {
        /// The task variable.
        var: VarId,
        /// Abstract state per value of `var`; `None` if pruned.
        table: Vec<Option<usize>>,
    },
    /// The product of two earlier nodes.
    Product {
        /// Index of the left child, whose state is the major index.
        left: usize,
        /// Index of the right child.
        right: usize,
        /// Number of abstract states of the right child.
        right_size: usize,
        /// Abstract state per `left * right_size + right`; `None` if pruned.
        table: Vec<Option<usize>>,
    },
}

/// A node's bounds during one evaluation.
struct Evaluated {
    /// The cheapest price bound per abstract state of the node.
    cost: Vec<u64>,
    /// The factors the node's variables touch, sorted: leaf indices, and
    /// `usize::MAX` for the center.
    factors: Vec<usize>,
}

/// A merge-and-shrink heuristic adapted to decoupled states; admissible, so
/// it can drive [`crate::astar`].
///
/// A merge-and-shrink abstraction maps each explicit state through a merge
/// tree of lookup tables to an abstract state with an admissible goal
/// distance. A decoupled state stands for many explicit states at once, so
/// the adapter evaluates the tree on [`FactPrices`] instead of on one state:
/// bottom-up, every tree node gets the cheapest *price bound* of each of its
/// abstract states, and the estimate is the cheapest `bound + distance` at
/// the root.
///
/// # Admissibility
///
/// For a tree node `n`, let `F(n)` be the factors (the leaves, and the
/// center as one more) that its variables touch, and for a member state
/// `s` let `P_n(s)` be the sum of the prices `s` pays in the factors of
/// `F(n)`. By induction, `cost_n(a) <= P_n(s)` for every member state `s`
/// that `n` maps to `a`:
///
/// - at a variable `v`, `cost(a)` is the cheapest price of a fact `v = x`
///   with `x` mapped to `a`, and the price of `v = s[v]` is at most what `s`
///   pays in `v`'s factor, by the contract of [`FactPrices`];
/// - at a product of children `l` and `r` that touch disjoint sets of
///   factors, `P_n = P_l + P_r`, so the sum of the children's bounds is a
///   bound;
/// - if they share a factor, its price must not be counted twice (two
///   variables of one leaf are bought with one leaf state), and
///   `P_n >= max(P_l, P_r)` makes the larger bound a bound.
///
/// At the root `F` contains every factor, so a member state `s` mapped to
/// `a` costs at least `cost(a)` to reach and at least `dist(a)` (the
/// abstraction is admissible) to complete; the minimum over all abstract
/// states is a lower bound on the cheapest completion through any member
/// state. Pruned abstract states (`None` in a table) are dead ends or
/// unreachable from the initial state; every member state of a decoupled
/// state is reachable, so skipping them is safe.
///
/// # Construction
///
/// [`merge_and_shrink::MergeAndShrinkHeuristic`] keeps its merge tree
/// private, so [`MasHeuristic::build`] re-runs the merge loop from the
/// public parts of [`merge_and_shrink`]. Label reduction is not public
/// either, so [`MergeAndShrink::label_reduction`] is ignored; that only
/// weakens shrinking, never admissibility.
///
/// # Examples
///
/// ```
/// use decoupled_search::{Factoring, MasHeuristic, astar};
/// use merge_and_shrink::{Bisimulation, LinearMerge, MergeAndShrink};
/// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
///
/// let op = |pre: Vec<Fact>, eff: Fact| Operator {
///     name: "op".into(),
///     pre,
///     eff: vec![Effect::new(eff)],
///     cost: 2,
/// };
/// let task = Task {
///     variables: vec![Variable::new("truck", 2), Variable::new("pkg", 3)],
///     operators: vec![
///         op(vec![Fact::new(0, 0)], Fact::new(0, 1)),
///         op(vec![Fact::new(0, 0), Fact::new(1, 0)], Fact::new(1, 2)),
///         op(vec![Fact::new(0, 1), Fact::new(1, 2)], Fact::new(1, 1)),
///     ],
///     initial: State(vec![0, 0]),
///     goal: vec![Fact::new(1, 1)],
///     axioms: vec![],
/// };
///
/// let merge = LinearMerge::in_variable_order(task.num_variables());
/// let h = MasHeuristic::build(&task, MergeAndShrink::new(merge, Bisimulation));
/// let plan = astar(&task, &Factoring::fork(&task), &h)?.expect("solvable");
/// assert_eq!(plan.cost, 6);
/// # Ok::<(), decoupled_search::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct MasHeuristic {
    /// The merge tree, children before parents; the last node is the root.
    nodes: Vec<Node>,
    /// The goal distance per abstract state of the root; `None` for dead
    /// ends.
    distances: Vec<Option<u64>>,
}

impl MasHeuristic {
    /// Builds the merge-and-shrink abstraction of `task` with the merge
    /// strategy, shrink strategy, size limit and pruning of `config`, as
    /// [`MergeAndShrink::build`] would, except that label reduction is not
    /// applied (see the module documentation).
    ///
    /// # Panics
    ///
    /// Panics if `task` has no variables, if `config.max_states` is 0, or if
    /// the merge strategy picks the same factor twice or one already merged.
    #[must_use]
    pub fn build<M: MergeStrategy, S: ShrinkStrategy>(
        task: &Task,
        mut config: MergeAndShrink<M, S>,
    ) -> Self {
        assert!(config.max_states > 0, "max_states must be at least 1");
        let costs = task.costs();
        let mut factors: Vec<Option<Factor>> = (0..task.num_variables())
            .map(|var| {
                let mut factor = Factor::atomic(task, var);
                if config.prune {
                    factor.prune(&costs);
                }

                Some(factor)
            })
            .collect();

        for _ in 1..factors.len() {
            let product = merge_next(&mut config, &mut factors, &costs);
            factors.push(Some(product));
        }

        let factor = factors
            .into_iter()
            .flatten()
            .next()
            .expect("the task has at least one variable");
        Self::from_factor(task, &factor)
    }

    /// The heuristic of a finished merge-and-shrink factor over all
    /// variables of `task`, whose labels are `task`'s operator ids.
    #[must_use]
    pub fn from_factor(task: &Task, factor: &Factor) -> Self {
        let mut nodes = Vec::new();
        flatten(&factor.tree, &mut nodes);

        Self {
            nodes,
            distances: factor.lts.goal_distances(&task.costs()),
        }
    }

    /// The number of abstract states of the root factor.
    #[must_use]
    pub fn num_abstract_states(&self) -> usize {
        self.distances.len()
    }

    /// Evaluates node `index` from its children in `done`.
    fn evaluate(&self, index: usize, done: &[Evaluated], facts: &FactPrices<'_>) -> Evaluated {
        match &self.nodes[index] {
            Node::Var { var, table } => {
                let mut cost = vec![INFINITE; size_of(table)];
                for (value, state) in table.iter().enumerate() {
                    let fact = Fact::new(*var, u32::try_from(value).expect("values fit u32"));
                    if let (Some(state), Some(price)) = (state, facts.get(fact)) {
                        cost[*state] = cost[*state].min(price);
                    }
                }

                Evaluated {
                    cost,
                    factors: vec![facts.leaf_of(*var).unwrap_or(usize::MAX)],
                }
            }
            Node::Product {
                left,
                right,
                right_size,
                table,
            } => product(&done[*left], &done[*right], *right_size, table),
        }
    }
}

impl DecoupledHeuristic for MasHeuristic {
    fn estimate(&self, facts: &FactPrices<'_>) -> Option<u64> {
        let mut done: Vec<Evaluated> = Vec::with_capacity(self.nodes.len());
        for index in 0..self.nodes.len() {
            let evaluated = self.evaluate(index, &done, facts);
            done.push(evaluated);
        }

        let root = done.last().expect("the tree has a node");
        root.cost
            .iter()
            .zip(&self.distances)
            .filter(|&(&cost, _)| cost != INFINITE)
            .filter_map(|(&cost, &distance)| distance.map(|d| cost.saturating_add(d)))
            .min()
    }
}

/// The bounds of a product node with children `l` and `r`: sums of the
/// children's bounds if they touch disjoint factors, maxima otherwise.
fn product(l: &Evaluated, r: &Evaluated, right_size: usize, table: &[Option<usize>]) -> Evaluated {
    let disjoint = !l.factors.iter().any(|f| r.factors.binary_search(f).is_ok());
    let mut cost = vec![INFINITE; size_of(table)];
    for (index, state) in table.iter().enumerate() {
        let Some(state) = *state else {
            continue;
        };

        let (a, b) = (l.cost[index / right_size], r.cost[index % right_size]);
        if a == INFINITE || b == INFINITE {
            continue;
        }

        let combined = if disjoint {
            a.saturating_add(b)
        } else {
            a.max(b)
        };
        cost[state] = cost[state].min(combined);
    }

    let mut factors = [l.factors.as_slice(), r.factors.as_slice()].concat();
    factors.sort_unstable();
    factors.dedup();

    Evaluated { cost, factors }
}

/// The number of abstract states a table maps into.
fn size_of(table: &[Option<usize>]) -> usize {
    table.iter().flatten().max().map_or(0, |&s| s + 1)
}

/// Appends `tree` to `nodes` in post-order and returns the index of its root.
fn flatten(tree: &MergeTree, nodes: &mut Vec<Node>) -> usize {
    let node = match tree {
        MergeTree::Leaf { var, table } => Node::Var {
            var: *var,
            table: table.clone(),
        },
        MergeTree::Node {
            left,
            right,
            right_size,
            table,
        } => Node::Product {
            left: flatten(left, nodes),
            right: flatten(right, nodes),
            right_size: *right_size,
            table: table.clone(),
        },
    };
    nodes.push(node);
    nodes.len() - 1
}

/// Takes the two factors the merge strategy picks, shrinks both so that
/// their product fits the size limit, and returns their product, pruned if
/// enabled.
///
/// # Panics
///
/// Panics if the merge strategy picks the same factor twice or an inactive
/// one.
fn merge_next<M: MergeStrategy, S: ShrinkStrategy>(
    config: &mut MergeAndShrink<M, S>,
    factors: &mut [Option<Factor>],
    costs: &[u64],
) -> Factor {
    let (i, j) = config.merge.next_pair(factors, costs);
    assert_ne!(i, j, "merge strategy returned the same factor twice");
    let mut a = factors[i]
        .take()
        .expect("merge strategy chose an inactive factor");
    let mut b = factors[j]
        .take()
        .expect("merge strategy chose an inactive factor");

    let (max_a, max_b) = shrink_sizes(a.lts.num_states, b.lts.num_states, config.max_states);
    for (factor, max) in [(&mut a, max_a), (&mut b, max_b)] {
        if let Some((map, n)) = config.shrink.compute(factor, costs, max) {
            factor.apply_abstraction(&map, n);
        }
    }

    let mut product = Factor::product(a, b);
    if config.prune {
        product.prune(costs);
    }

    product
}

/// Size limits for two factors of sizes `a` and `b` so that their product
/// has at most `max_states` states, as in [`merge_and_shrink`]: each keeps
/// its size if the product fits; otherwise the smaller one keeps as much as
/// it can and the larger one gets the rest, and if neither fits within
/// `isqrt(max_states)`, both get about that.
fn shrink_sizes(a: usize, b: usize, max_states: usize) -> (usize, usize) {
    let (a, b) = (a.min(max_states), b.min(max_states));
    if a.saturating_mul(b) <= max_states {
        return (a, b);
    }

    let balanced = max_states.isqrt();
    if a <= balanced {
        (a, max_states / a)
    } else if b <= balanced {
        (max_states / b, b)
    } else {
        (balanced, max_states / balanced)
    }
}
