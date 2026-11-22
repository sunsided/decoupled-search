//! The causal graph as decoupled search sees it, with derived variables
//! replaced by the state variables they depend on.
//!
//! The search attaches a derived variable to the one leaf its inputs lie in
//! (or to the center), so an operator that reads it reads those inputs. The
//! plain causal graph routes such interactions through the derived vertex;
//! a strategy that splits the graph there would put two leaves in contact,
//! or a derived variable's inputs in two leaves, which the search rejects
//! ([`crate::Error::AxiomAcrossLeaves`]). Every strategy and
//! [`Factoring::check_star`](super::Factoring::check_star) works on this
//! graph instead. Without axioms it is the causal graph.

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{CausalGraph, Task, VarId};

/// The causal graph over the state variables of a task, derived vertices
/// contracted into their inputs.
///
/// - An arc `v -> w` of the causal graph between state variables is kept.
/// - An arc `d -> w` from a derived variable `d` to a state variable `w`
///   becomes `s -> w` for every state variable `s` that `d` depends on.
/// - The state variables a derived variable depends on are joined by arcs
///   both ways, so every strategy keeps them in one factor.
///
/// Derived variables have no arcs; strategies leave them in the center.
#[derive(Clone, Debug)]
pub(super) struct FactorGraph {
    /// Per variable, its successors, sorted and without duplicates or
    /// self-loops.
    successors: Vec<Vec<VarId>>,
    /// Per variable, its predecessors, sorted and without duplicates or
    /// self-loops.
    predecessors: Vec<Vec<VarId>>,
}

impl FactorGraph {
    /// Builds the graph of `task`.
    pub(super) fn new(task: &Task) -> Self {
        let n = task.num_variables();
        let graph = CausalGraph::new(task);
        let inputs = derived_inputs(task);
        let mut successors = vec![Vec::new(); n];
        for v in 0..n {
            let sources = if task.variables[v].is_derived() {
                inputs[v].as_slice()
            } else {
                core::slice::from_ref(&v)
            };
            for &w in graph.successors(v) {
                if !task.variables[w].is_derived() {
                    for &s in sources {
                        successors[s].push(w);
                    }
                }
            }

            for &s in &inputs[v] {
                successors[s].extend_from_slice(&inputs[v]);
            }
        }

        let mut predecessors = vec![Vec::new(); n];
        for (v, list) in successors.iter_mut().enumerate() {
            list.sort_unstable();
            list.dedup();
            list.retain(|&w| w != v);
            for &w in list.iter() {
                predecessors[w].push(v);
            }
        }

        Self {
            successors,
            predecessors,
        }
    }

    /// The successors of `v`, sorted.
    pub(super) fn successors(&self, v: VarId) -> &[VarId] {
        &self.successors[v]
    }

    /// The predecessors of `v`, sorted.
    pub(super) fn predecessors(&self, v: VarId) -> &[VarId] {
        &self.predecessors[v]
    }
}

/// Per variable, the state variables it depends on through axioms, directly
/// or through other derived variables; empty for state variables.
fn derived_inputs(task: &Task) -> Vec<Vec<VarId>> {
    let n = task.num_variables();
    let mut direct = vec![Vec::new(); n];
    for axiom in &task.axioms {
        direct[axiom.head.var].extend(axiom.conditions.iter().map(|f| f.var));
    }

    (0..n)
        .map(|derived| {
            if !task.variables[derived].is_derived() {
                return Vec::new();
            }

            let mut seen = vec![false; n];
            seen[derived] = true;
            let mut stack = vec![derived];
            let mut inputs = Vec::new();
            while let Some(v) = stack.pop() {
                for &u in &direct[v] {
                    if core::mem::replace(&mut seen[u], true) {
                        continue;
                    }

                    if task.variables[u].is_derived() {
                        stack.push(u);
                    } else {
                        inputs.push(u);
                    }
                }
            }

            inputs
        })
        .collect()
}
