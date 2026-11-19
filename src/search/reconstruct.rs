//! Turns a path of decoupled states into a sequential plan.
//!
//! The search stores only global actions. Leaf paths are recomputed here,
//! stage by stage and backwards, for the leaf states the plan ends in.

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::OpId;

use crate::Plan;
use crate::decoupled::{DecoupledState, DecoupledTask, Leaf, Price};

/// A path of decoupled states, initial first, each with the index of the
/// global action that led to it (`None` for the initial state).
type Path<'a> = [(Option<usize>, &'a DecoupledState)];

/// The sequential plan for a solution ending in the last state of `path`
/// at total cost `cost`.
///
/// The plan runs, per stage, the leaf actions of every leaf, then the next
/// global action. Leaf actions of different leaves commute within a stage,
/// since each reads only the fixed center and its own leaf.
pub(super) fn plan(task: &DecoupledTask, path: &Path<'_>, cost: Price) -> Plan {
    let leaf_ops: Vec<Vec<Vec<OpId>>> = (0..task.leaves.len())
        .map(|leaf| leaf_stages(task, path, leaf))
        .collect();

    let mut ops = Vec::new();
    for (stage, (action, _)) in path.iter().enumerate() {
        if let Some(action) = action {
            ops.push(task.globals[*action].op);
        }

        for stages in &leaf_ops {
            ops.extend_from_slice(&stages[stage]);
        }
    }

    Plan { ops, cost }
}

/// The leaf actions of leaf `leaf` in every stage of `path`, ending in the
/// cheapest goal leaf state of the last stage.
fn leaf_stages(task: &DecoupledTask, path: &Path<'_>, leaf: usize) -> Vec<Vec<OpId>> {
    let compiled = &task.leaves[leaf];
    let (_, last) = path.last().expect("a path has an initial state");
    let (_, mut target) = compiled
        .cheapest_goal(&last.prices[leaf])
        .expect("a solution reaches a leaf goal state");

    let mut stages = vec![Vec::new(); path.len()];
    for stage in (0..path.len()).rev() {
        let seeds = stage_seeds(task, path, leaf, stage);
        let mut prices = seeds.clone();
        let steps = compiled.close(&path[stage].1.center, &mut prices);

        let mut rank = target;
        while let Some((from, op)) = steps[rank] {
            stages[stage].push(op);
            rank = from;
        }

        stages[stage].reverse();
        if stage > 0 {
            target = step_back(task, path, leaf, stage, rank, seeds[rank]);
        }
    }

    stages
}

/// The prices of leaf `leaf` at the start of stage `stage`, before leaf
/// reachability: the initial leaf state for stage 0, otherwise the previous
/// stage's prices carried across the global action.
fn stage_seeds(task: &DecoupledTask, path: &Path<'_>, leaf: usize, stage: usize) -> Vec<Price> {
    let compiled: &Leaf = &task.leaves[leaf];
    let Some(action) = path[stage].0 else {
        return compiled.initial_prices();
    };

    let previous = &path[stage - 1].1.prices[leaf];
    match &task.globals[action].parts[leaf] {
        Some(part) => compiled.transfer(part, previous),
        None => previous.clone(),
    }
}

/// The leaf state in stage `stage - 1` that the global action into `stage`
/// carried onto leaf state `seed` at price `price`.
fn step_back(
    task: &DecoupledTask,
    path: &Path<'_>,
    leaf: usize,
    stage: usize,
    seed: usize,
    price: Price,
) -> usize {
    let action = path[stage].0.expect("only the initial stage has no action");
    match &task.globals[action].parts[leaf] {
        Some(part) => {
            task.leaves[leaf].transfer_source(part, &path[stage - 1].1.prices[leaf], seed, price)
        }
        None => seed,
    }
}
