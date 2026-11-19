//! Hand-encoded gripper fixtures with known optimal costs.

use decoupled_search::Factoring;
use sas_plus::Task;
use sas_plus::fdr::SasFile;

use crate::checked_cost;
use crate::dijkstra::optimal_cost;

/// The task in `tests/fixtures/<name>/output.sas`.
fn fixture(name: &str) -> Task {
    let path = format!(
        "{}/tests/fixtures/{name}/output.sas",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture exists");
    SasFile::parse(&text).expect("fixture parses").to_task()
}

#[test]
fn gripper_optimal_costs() {
    for (name, optimum) in [("gripper-costs", 7), ("gripper-unit", 5)] {
        let task = fixture(name);
        assert_eq!(optimal_cost(&task), Some(optimum), "{name}");

        let fork = Factoring::fork(&task);
        assert_eq!(fork.center(), &[0], "{name}: the robot is the center");
        assert_eq!(checked_cost(&task, &fork), Some(optimum), "{name}");

        // Balls as leaves: picks and drops also change the grippers, so they become global.
        let balls = Factoring::new(&task, vec![vec![1], vec![2]]).expect("valid");
        assert_eq!(checked_cost(&task, &balls), Some(optimum), "{name}");
    }
}
