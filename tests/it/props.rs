//! Property tests: decoupled search under every factoring strategy agrees
//! with explicit Dijkstra, and the admissible heuristics never overestimate
//! on explicit states. proptest shrinks a failure to a small seed and task.

use decoupled_search::{DecoupledHeuristic, FactPrices, HMax};
use proptest::prelude::*;
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{State, Task};

use crate::checked_cost;
use crate::dijkstra::optimal_cost;
use crate::star::{random_star_task, strategies};

/// Random-task parameters: up to five variables, conditional effects, zero
/// costs, and (when `derived`) up to two derived variables.
fn params(derived: bool) -> impl Strategy<Value = RandomTaskParams> {
    let num_derived = if derived { 0..=2_usize } else { 0..=0 };

    (
        1..=5_usize,
        2..=3_u32,
        1..=10_usize,
        0..=5_u32,
        0..=2_usize,
        num_derived,
    )
        .prop_map(
            |(num_vars, max_domain, num_ops, max_cost, max_conditions, num_derived)| {
                RandomTaskParams {
                    num_vars,
                    max_domain,
                    num_ops,
                    max_cost,
                    max_conditions,
                    num_derived,
                }
            },
        )
}

/// A random task and an arbitrary state over its variables, as raw values
/// reduced modulo each domain size.
fn task_and_state() -> impl Strategy<Value = (Task, State)> {
    (
        any::<u64>(),
        params(false),
        prop::collection::vec(any::<u32>(), 5),
    )
        .prop_map(|(seed, params, raw)| {
            let task = random_task(seed, params);
            let values = task
                .variables
                .iter()
                .zip(raw)
                .map(|(var, value)| value % var.domain_size)
                .collect();

            (task, State(values))
        })
}

/// The admissible heuristics under test: `h^max`, and LM-cut in both
/// evaluation modes with the `lm-cut` feature.
fn admissible(task: &Task) -> Vec<Box<dyn DecoupledHeuristic>> {
    let mut all: Vec<Box<dyn DecoupledHeuristic>> = vec![Box::new(HMax::new(task))];
    #[cfg(feature = "lm-cut")]
    {
        all.push(Box::new(decoupled_search::LmCutHeuristic::new(task)));
        all.push(Box::new(
            decoupled_search::LmCutHeuristic::with_price_levels(task),
        ));
    }

    all
}

proptest! {
    /// Every factoring strategy (fork, inverted fork, X-shape, both strict
    /// stars and the automatic choice) gives the optimal cost on random
    /// tasks with conditional effects, axioms and zero costs, under every
    /// cost-optimal configuration, or rejects the task.
    #[test]
    fn strategies_match_dijkstra_on_random_tasks(seed in any::<u64>(), params in params(true)) {
        let task = random_task(seed, params);
        let expected = optimal_cost(&task);
        for (name, factoring) in strategies(&task) {
            if decoupled_search::search(&task, &factoring).is_err() {
                continue;
            }

            prop_assert_eq!(checked_cost(&task, &factoring), expected, "{}", name);
        }
    }

    /// On an arbitrary explicit state, each admissible heuristic is at most
    /// the optimal cost from that state, and reports a dead end (`None`)
    /// only where no plan exists.
    #[test]
    fn admissible_heuristics_never_overestimate((task, state) in task_and_state()) {
        let from_state = Task { initial: state.clone(), ..task.clone() };
        let optimal = optimal_cost(&from_state);
        let prices = FactPrices::from_state(&task, &state);
        for heuristic in admissible(&task) {
            match (heuristic.estimate(&prices), optimal) {
                (Some(h), Some(cost)) => prop_assert!(h <= cost, "h = {} > {}", h, cost),
                (None, Some(cost)) => prop_assert!(false, "dead end, but a plan costs {}", cost),
                (_, None) => {}
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: ProptestConfig::default().cases / 8, ..ProptestConfig::default() })]

    /// The same on random star tasks, where every strategy finds leaves.
    /// These tasks are larger, so the run uses an eighth of the cases
    /// (`PROPTEST_CASES` still scales it).
    #[test]
    fn strategies_match_dijkstra_on_star_tasks(seed in any::<u64>()) {
        let task = random_star_task(seed);
        let expected = optimal_cost(&task);
        for (name, factoring) in strategies(&task) {
            prop_assert_eq!(checked_cost(&task, &factoring), expected, "{}", name);
        }
    }
}
