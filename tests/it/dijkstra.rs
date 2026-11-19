//! Plain explicit-state Dijkstra over `sas_plus` states, the reference for
//! optimal costs.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use sas_plus::{State, Task};

/// The optimal plan cost of `task`, or `None` if it is unsolvable.
pub fn optimal_cost(task: &Task) -> Option<u64> {
    let initial = task.initial_state();
    let mut best: HashMap<State, u64> = HashMap::from([(initial.clone(), 0)]);
    // `State` is not `Ord`, so the heap holds the raw values.
    let mut open = BinaryHeap::from([Reverse((0, initial.0))]);

    while let Some(Reverse((g, values))) = open.pop() {
        let state = State(values);
        if best[&state] < g {
            continue;
        }

        if task.is_goal(&state) {
            return Some(g);
        }

        for op in task.applicable_operators(&state) {
            let next = task.apply(op, &state);
            let cost = g + u64::from(task.operators[op].cost);
            if best.get(&next).is_none_or(|&old| cost < old) {
                best.insert(next.clone(), cost);
                open.push(Reverse((cost, next.0)));
            }
        }
    }

    None
}
