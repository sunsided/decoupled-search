//! Fuzzes [`Plan::validate`] on arbitrary operator sequences of small random
//! tasks with conditional effects and axioms: it never panics and accepts
//! exactly the plans [`sas_plus::Task::validate_plan`] accepts, at the cost
//! that one computes.

#![no_main]
#![forbid(unsafe_code)]

use decoupled_search::Plan;
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;
use sas_plus::random::{RandomTaskParams, random_task};

/// The decoded fuzz input: a random task and a candidate plan for it.
#[derive(Arbitrary, Debug)]
struct Input {
    /// Seed of [`random_task`].
    seed: u64,
    /// Operator ids, reduced modulo one more than the operator count, so
    /// unknown operators occur too.
    ops: Vec<u8>,
    /// The claimed plan cost; replaced by the true cost on every other
    /// input so that valid plans are reached.
    cost: u64,
    /// Whether to claim the true cost instead of [`Input::cost`].
    true_cost: bool,
}

fuzz_target!(|input: Input| {
    let params = RandomTaskParams {
        max_conditions: 2,
        num_derived: 2,
        ..RandomTaskParams::default()
    };
    let task = random_task(input.seed, params);
    let bound = task.operators.len() + 1;
    let ops: Vec<usize> = input
        .ops
        .iter()
        .map(|&op| usize::from(op) % bound)
        .collect();
    let expected = ops
        .iter()
        .all(|&op| op < task.operators.len())
        .then(|| task.validate_plan(&ops))
        .flatten();
    let cost = match (input.true_cost, expected) {
        (true, Some(cost)) => cost,
        _ => input.cost,
    };

    let result = Plan { ops, cost }.validate(&task);
    assert_eq!(result.is_ok(), expected == Some(cost), "{result:?}");
});
