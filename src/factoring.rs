//! Center/leaf partitions of a task's variables, and strategies that derive
//! them from the causal graph.

mod scoring;
mod star;

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{CausalGraph, Task, VarId};

pub use scoring::{FactoringScore, Objective, Strategy};

use crate::{Error, MAX_LEAF_STATES};

/// A partition of a task's variables into one center and any number of
/// leaves.
///
/// Every partition is valid: operators that would break the star topology
/// (for example by reading one leaf and writing another) are searched
/// explicitly as global actions. The factoring only decides how much of the
/// task is priced in the leaves instead of searched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Factoring {
    /// Center variables, sorted ascending.
    center: Vec<VarId>,
    /// Leaf variables, each leaf non-empty and with at most
    /// [`MAX_LEAF_STATES`] states. No variable appears twice.
    leaves: Vec<Vec<VarId>>,
}

impl Factoring {
    /// A factoring with the given leaves; all other variables form the center.
    ///
    /// # Errors
    ///
    /// - [`Error::UnknownVariable`] if a leaf names a variable `task` lacks.
    /// - [`Error::DuplicateVariable`] if a variable appears twice.
    /// - [`Error::DerivedInLeaf`] if a leaf names a derived variable.
    /// - [`Error::EmptyLeaf`] if a leaf has no variables.
    /// - [`Error::LeafTooLarge`] if a leaf has more than [`MAX_LEAF_STATES`]
    ///   states.
    pub fn new(task: &Task, leaves: Vec<Vec<VarId>>) -> Result<Self, Error> {
        let mut in_leaf = vec![false; task.num_variables()];
        for (index, leaf) in leaves.iter().enumerate() {
            if leaf.is_empty() {
                return Err(Error::EmptyLeaf { leaf: index });
            }

            for &var in leaf {
                let seen = in_leaf.get_mut(var).ok_or(Error::UnknownVariable { var })?;
                if *seen {
                    return Err(Error::DuplicateVariable { var });
                }

                if task.variables[var].is_derived() {
                    return Err(Error::DerivedInLeaf { var });
                }

                *seen = true;
            }

            if leaf_states(task, leaf).is_none() {
                return Err(Error::LeafTooLarge { leaf: index });
            }
        }

        Ok(Self::from_leaves(task, leaves))
    }

    /// A factoring with the given, already checked leaves; all other
    /// variables form the center. Derived variables are moved to the center,
    /// and leaves left empty by that are dropped.
    fn from_leaves(task: &Task, mut leaves: Vec<Vec<VarId>>) -> Self {
        for leaf in &mut leaves {
            leaf.retain(|&v| !task.variables[v].is_derived());
        }

        leaves.retain(|leaf| !leaf.is_empty());
        let mut in_leaf = vec![false; task.num_variables()];
        for &v in leaves.iter().flatten() {
            in_leaf[v] = true;
        }

        let center = (0..task.num_variables()).filter(|&v| !in_leaf[v]).collect();
        Self { center, leaves }
    }

    /// The fork factoring: every sink strongly connected component of the
    /// causal graph (one no arc leaves) with at most [`MAX_LEAF_STATES`]
    /// states is a leaf, everything else is the center.
    ///
    /// Operators on a sink component read only the center and that component,
    /// so all of them are priced in the leaf. A task whose causal graph is one
    /// small component becomes one leaf with an empty center.
    #[must_use]
    pub fn fork(task: &Task) -> Self {
        let graph = CausalGraph::new(task);
        Self::from_components(
            task,
            sink_components(task.num_variables(), |v| graph.successors(v)),
        )
    }

    /// The inverted-fork factoring: every source strongly connected component
    /// of the causal graph (one no arc enters) with at most
    /// [`MAX_LEAF_STATES`] states is a leaf, everything else is the center.
    ///
    /// Operators on a source component read and write only that component,
    /// and the center may read it. Such operators are leaf actions unless they
    /// also touch the center.
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::Factoring;
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A switch (variable 0) that a lamp (variable 1) depends on.
    /// let op = |pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let task = Task {
    ///     variables: vec![Variable::new("switch", 2), Variable::new("lamp", 2)],
    ///     operators: vec![
    ///         op(vec![], Fact::new(0, 1)),
    ///         op(vec![Fact::new(0, 1)], Fact::new(1, 1)),
    ///     ],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let factoring = Factoring::inverted_fork(&task);
    /// assert_eq!(factoring.leaves(), &[vec![0]]);
    /// assert_eq!(factoring.center(), &[1]);
    /// ```
    #[must_use]
    pub fn inverted_fork(task: &Task) -> Self {
        let graph = CausalGraph::new(task);
        Self::from_components(
            task,
            sink_components(task.num_variables(), |v| graph.predecessors(v)),
        )
    }

    /// The X-shape factoring (Gnad, Hoffmann and Domshlak, `SoCS` 2015): the
    /// fork factoring, plus every small source strongly connected component
    /// of the rest of the causal graph that has no arc into a fork leaf.
    ///
    /// The paper runs the fork strategy and then the inverted-fork strategy
    /// on the fork center, putting back into the center every inverted-fork
    /// leaf with an arc into a fork leaf. Fork leaves have no outgoing arcs,
    /// so the source components of the fork center are source components
    /// of the whole causal graph. Every leaf then touches only the center:
    /// fork leaves receive arcs from it, inverted-fork leaves send arcs to
    /// it, so the result passes [`Factoring::check_star`]. Leaves keep the
    /// [`MAX_LEAF_STATES`] bound of the other strategies.
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::Factoring;
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // A switch (0) enables a truck (1) that moves a package (2).
    /// let op = |pre: Vec<Fact>, eff: Fact| Operator {
    ///     name: "op".into(),
    ///     pre,
    ///     eff: vec![Effect::new(eff)],
    ///     cost: 1,
    /// };
    /// let var = |name: &str| Variable::new(name, 2);
    /// let task = Task {
    ///     variables: vec![var("switch"), var("truck"), var("package")],
    ///     operators: vec![
    ///         op(vec![], Fact::new(0, 1)),
    ///         op(vec![Fact::new(0, 1)], Fact::new(1, 1)),
    ///         op(vec![Fact::new(1, 1)], Fact::new(2, 1)),
    ///     ],
    ///     initial: State(vec![0, 0, 0]),
    ///     goal: vec![Fact::new(2, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let factoring = Factoring::x_shape(&task);
    /// assert_eq!(factoring.center(), &[1]);
    /// assert_eq!(factoring.leaves(), &[vec![2], vec![0]]);
    /// assert_eq!(factoring.check_star(&task), Ok(()));
    /// ```
    #[must_use]
    pub fn x_shape(task: &Task) -> Self {
        let n = task.num_variables();
        let graph = CausalGraph::new(task);
        let mut leaves: Vec<Vec<VarId>> = sink_components(n, |v| graph.successors(v))
            .into_iter()
            .filter(|c| leaf_states(task, c).is_some())
            .collect();
        let mut in_fork_leaf = vec![false; n];
        for &v in leaves.iter().flatten() {
            in_fork_leaf[v] = true;
        }

        let inverted = sink_components(n, |v| graph.predecessors(v))
            .into_iter()
            .filter(|c| c.iter().all(|&v| !in_fork_leaf[v]))
            .filter(|c| leaf_states(task, c).is_some())
            .filter(|c| {
                c.iter()
                    .all(|&v| graph.successors(v).iter().all(|&u| !in_fork_leaf[u]))
            });
        leaves.extend(inverted);

        Self::from_leaves(task, leaves)
    }

    /// A factoring whose leaves are the given disjoint, non-empty components
    /// that have at most [`MAX_LEAF_STATES`] states; larger components join
    /// the center.
    fn from_components(task: &Task, components: Vec<Vec<VarId>>) -> Self {
        let leaves = components
            .into_iter()
            .filter(|c| leaf_states(task, c).is_some())
            .collect();

        Self::from_leaves(task, leaves)
    }

    /// Checks that the factoring is a strict star: no causal-graph arc
    /// connects variables of two different leaves, so leaves interact only
    /// through the center.
    ///
    /// Search does not require this, since an operator that couples two
    /// leaves becomes a global action. In a star factoring no operator writes
    /// one leaf while reading or writing another, which keeps the leaves
    /// independent. An operator may still read several leaves and write only
    /// the center. All strategies of this type except
    /// [`Factoring::new`] produce star factorings.
    ///
    /// # Errors
    ///
    /// [`Error::NotStar`] with the first offending arc `from -> to` (in
    /// variable order), and [`Error::FactoringMismatch`] if the factoring was
    /// built for a task with fewer variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Error, Factoring};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// // One operator reads variable 0 and writes variable 1.
    /// let task = Task {
    ///     variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
    ///     operators: vec![Operator {
    ///         name: "op".into(),
    ///         pre: vec![Fact::new(0, 1)],
    ///         eff: vec![Effect::new(Fact::new(1, 1))],
    ///         cost: 1,
    ///     }],
    ///     initial: State(vec![0, 0]),
    ///     goal: vec![Fact::new(1, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// let split = Factoring::new(&task, vec![vec![0], vec![1]])?;
    /// assert_eq!(split.check_star(&task), Err(Error::NotStar { from: 0, to: 1 }));
    /// assert_eq!(Factoring::fork(&task).check_star(&task), Ok(()));
    /// # Ok::<(), Error>(())
    /// ```
    pub fn check_star(&self, task: &Task) -> Result<(), Error> {
        let mut leaf_of = vec![None; task.num_variables()];
        for (leaf, vars) in self.leaves.iter().enumerate() {
            for &v in vars {
                *leaf_of.get_mut(v).ok_or(Error::FactoringMismatch)? = Some(leaf);
            }
        }

        let graph = CausalGraph::new(task);
        for from in 0..task.num_variables() {
            let Some(leaf) = leaf_of[from] else {
                continue;
            };

            if let Some(&to) = graph
                .successors(from)
                .iter()
                .find(|&&to| leaf_of[to].is_some_and(|other| other != leaf))
            {
                return Err(Error::NotStar { from, to });
            }
        }

        Ok(())
    }

    /// The leaf of every variable of `task`, `None` for center variables.
    ///
    /// # Errors
    ///
    /// [`Error::FactoringMismatch`] if the factoring does not partition
    /// exactly the variables of `task`.
    pub(crate) fn leaf_of(&self, task: &Task) -> Result<Vec<Option<usize>>, Error> {
        let mut leaf_of = vec![None; task.num_variables()];
        let mut seen = vec![false; task.num_variables()];
        let center = self.center.iter().map(|&v| (v, None));
        let leaves = self
            .leaves
            .iter()
            .enumerate()
            .flat_map(|(leaf, vars)| vars.iter().map(move |&v| (v, Some(leaf))));
        for (var, leaf) in center.chain(leaves) {
            let slot = seen.get_mut(var).ok_or(Error::FactoringMismatch)?;
            if core::mem::replace(slot, true) {
                return Err(Error::FactoringMismatch);
            }

            leaf_of[var] = leaf;
        }

        if seen.contains(&false) {
            return Err(Error::FactoringMismatch);
        }

        Ok(leaf_of)
    }

    /// The center variables, sorted ascending. May be empty.
    #[must_use]
    pub fn center(&self) -> &[VarId] {
        &self.center
    }

    /// The leaves, each a non-empty list of variables.
    #[must_use]
    pub fn leaves(&self) -> &[Vec<VarId>] {
        &self.leaves
    }
}

/// The number of states of a leaf over `vars`, or `None` if it exceeds
/// [`MAX_LEAF_STATES`].
fn leaf_states(task: &Task, vars: &[VarId]) -> Option<usize> {
    vars.iter().try_fold(1_usize, |size, &v| {
        let domain = usize::try_from(task.variables[v].domain_size).ok()?;
        size.checked_mul(domain).filter(|&s| s <= MAX_LEAF_STATES)
    })
}

/// The sink strongly connected components of the graph over `n` vertices
/// with arcs `v -> successors(v)`, each sorted, in order of their smallest
/// vertex.
///
/// A vertex's component is a sink iff every vertex it reaches reaches it back.
///
/// shortcut: one depth-first search per vertex, `O(n * (n + arcs))` time and
/// `O(n^2)` memory; switch to Tarjan if tasks with thousands of variables
/// show up.
fn sink_components<'g>(
    n: usize,
    successors: impl Fn(VarId) -> &'g [VarId] + Copy,
) -> Vec<Vec<VarId>> {
    let reach: Vec<Vec<bool>> = (0..n).map(|v| reachable(n, successors, v)).collect();
    let mut assigned = vec![false; n];
    let mut sinks = Vec::new();
    for v in 0..n {
        if assigned[v] {
            continue;
        }

        let reached: Vec<VarId> = (0..n).filter(|&u| reach[v][u]).collect();
        for &u in &reached {
            assigned[u] |= reach[u][v];
        }

        if reached.iter().all(|&u| reach[u][v]) {
            sinks.push(reached);
        }
    }

    sinks
}

/// The vertices reachable from `start` over arcs `v -> successors(v)`,
/// including `start`.
fn reachable<'g>(n: usize, successors: impl Fn(VarId) -> &'g [VarId], start: VarId) -> Vec<bool> {
    let mut seen = vec![false; n];
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(v) = stack.pop() {
        for &w in successors(v) {
            if !seen[w] {
                seen[w] = true;
                stack.push(w);
            }
        }
    }

    seen
}
