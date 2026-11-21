//! Fuzzes decoupled search on small random tasks under an arbitrary leaf
//! partition and the automatic factoring.
//!
//! Building a factoring and searching may reject the input with an
//! [`decoupled_search::Error`], but must never panic. Whatever is accepted
//! must agree with explicit search (the factoring without leaves):
//! uniform-cost search and A* with `h^max` and LM-cut find plans of the
//! optimal cost, greedy search finds a plan exactly when one exists, and
//! every plan passes [`Plan::validate`].

#![no_main]
#![forbid(unsafe_code)]

use decoupled_search::{
    Config, DecoupledHeuristic, Factoring, HFf, HMax, LmCutHeuristic, Plan, Pruning,
};
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;
use sas_plus::Task;
use sas_plus::random::{RandomTaskParams, random_task};

/// Every dominance criterion, indexed by [`Input::pruning`].
const PRUNINGS: [Pruning; 6] = [
    Pruning::Basic,
    Pruning::Frontier,
    Pruning::EffectivePrices,
    Pruning::Simulation,
    Pruning::EffectiveSimulation,
    Pruning::Adaptive,
];

/// The decoded fuzz input: a random task, a leaf partition and a search
/// configuration. Sizes are reduced modulo small bounds so that explicit
/// search stays cheap.
#[derive(Arbitrary, Debug)]
struct Input {
    /// Seed of [`random_task`].
    seed: u64,
    /// State variables, 1 to 5.
    num_vars: u8,
    /// Largest domain size, 2 or 3.
    max_domain: u8,
    /// Operators, 1 to 10.
    num_ops: u8,
    /// Largest operator cost, 0 to 5 (zero costs included).
    max_cost: u8,
    /// Conditions per conditional effect, 0 to 2.
    max_conditions: u8,
    /// Derived variables, 0 to 2.
    num_derived: u8,
    /// Leaves as raw variable indices, reduced modulo one more than the
    /// task's variable count, so unknown variables occur too. At most four
    /// leaves are used.
    leaves: Vec<Vec<u8>>,
    /// Index into [`PRUNINGS`], reduced modulo its length.
    pruning: u8,
    /// [`Config::fold_g`].
    fold_g: bool,
    /// [`Config::reduce_leaves`].
    reduce_leaves: bool,
}

impl Input {
    /// The random-task parameters, each reduced into its documented range.
    fn params(&self) -> RandomTaskParams {
        RandomTaskParams {
            num_vars: usize::from(self.num_vars % 5 + 1),
            max_domain: u32::from(self.max_domain % 2 + 2),
            num_ops: usize::from(self.num_ops % 10 + 1),
            max_cost: u32::from(self.max_cost % 6),
            max_conditions: usize::from(self.max_conditions % 3),
            num_derived: usize::from(self.num_derived % 3),
        }
    }

    /// The leaf partition over `task`'s variables, unchecked.
    fn leaves(&self, task: &Task) -> Vec<Vec<usize>> {
        let bound = task.num_variables() + 1;

        self.leaves
            .iter()
            .take(4)
            .map(|leaf| leaf.iter().map(|&v| usize::from(v) % bound).collect())
            .collect()
    }

    /// The search configuration.
    fn config(&self) -> Config {
        Config {
            pruning: PRUNINGS[usize::from(self.pruning) % PRUNINGS.len()],
            fold_g: self.fold_g,
            reduce_leaves: self.reduce_leaves,
            ..Config::default()
        }
    }
}

/// The cost of `plan` after checking it with [`Plan::validate`]; `None` if
/// there is no plan.
fn validated(task: &Task, plan: Option<Plan>) -> Option<u64> {
    let plan = plan?;
    assert_eq!(plan.validate(task), Ok(()), "plan {:?}", plan.ops);

    Some(plan.cost)
}

/// Checks every search under `factoring` against the optimal cost
/// `expected`. A factoring the search rejects is skipped.
fn check(task: &Task, factoring: &Factoring, config: Config, expected: Option<u64>) {
    let Ok(outcome) = config.search(task, factoring) else {
        return;
    };
    assert_eq!(
        validated(task, outcome.plan),
        expected,
        "uniform-cost search"
    );

    let lm_cut = LmCutHeuristic::new(task);
    let heuristics: [&dyn DecoupledHeuristic; 2] = [&HMax::new(task), &lm_cut];
    for heuristic in heuristics {
        let outcome = config
            .astar(task, factoring, heuristic)
            .expect("accepted by uniform-cost search");
        assert_eq!(validated(task, outcome.plan), expected, "A*");
    }

    let outcome = config
        .gbfs(task, factoring, &HFf::new(task))
        .expect("accepted by uniform-cost search");
    let cost = validated(task, outcome.plan);
    assert_eq!(cost.is_some(), expected.is_some(), "greedy solvability");
    assert!(
        cost >= expected,
        "greedy {cost:?} beats the optimum {expected:?}"
    );
}

fuzz_target!(|input: Input| {
    let task = random_task(input.seed, input.params());
    let explicit = Factoring::new(&task, Vec::new()).expect("no leaves is always valid");
    let expected = Config::default()
        .search(&task, &explicit)
        .expect("explicit search supports every valid task")
        .plan
        .map(|plan| plan.cost);
    let config = input.config();

    check(&task, &Factoring::auto(&task), config, expected);
    if let Ok(factoring) = Factoring::new(&task, input.leaves(&task)) {
        check(&task, &factoring, config, expected);
    }
});
