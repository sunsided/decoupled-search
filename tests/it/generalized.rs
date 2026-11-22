//! Generalized factorings (Gnad, Torralba and Fišer, ICAPS 2022) on their
//! Example 1: battery-powered agents that can charge each other.

use decoupled_search::{Config, Factoring, Pruning, Strategy};
use sas_plus::random::{RandomTaskParams, random_task};
use sas_plus::{Effect, Fact, Operator, State, Task, Variable};

use crate::checked_cost;
use crate::dijkstra::optimal_cost;

/// `n` agents on `m` locations in a line (moving costs one unit of
/// battery), batteries of capacity `b`. Agent 0 starts at location 0 with a
/// full battery, agent `i` at location `i` with an empty one; everyone must
/// reach location 0. Two agents at one location can pass one unit of
/// charge. Variables: agent `i` position `2i`, battery `2i + 1`.
fn agents(n: usize, m: u32, b: u32) -> Task {
    let pos = |i: usize| 2 * i;
    let bat = |i: usize| 2 * i + 1;
    let mut operators = Vec::new();
    for i in 0..n {
        for x in 0..m {
            for y in [x.wrapping_sub(1), x + 1].into_iter().filter(|&y| y < m) {
                for k in 1..=b {
                    operators.push(Operator {
                        name: format!("move-{i}-{x}-{y}-{k}"),
                        pre: vec![Fact::new(pos(i), x), Fact::new(bat(i), k)],
                        eff: vec![
                            Effect::new(Fact::new(pos(i), y)),
                            Effect::new(Fact::new(bat(i), k - 1)),
                        ],
                        cost: 1,
                    });
                }
            }
        }
        for j in (0..n).filter(|&j| j != i) {
            operators.extend(charges(i, j, m, b));
        }
    }

    let mut initial = Vec::new();
    for i in 0..n {
        initial.push(u32::try_from(i).expect("small"));
        initial.push(if i == 0 { b } else { 0 });
    }

    Task {
        variables: (0..n)
            .flat_map(|i| {
                [
                    Variable::new(format!("a{i}"), m),
                    Variable::new(format!("b{i}"), b + 1),
                ]
            })
            .collect(),
        operators,
        initial: State(initial),
        goal: (0..n).map(|i| Fact::new(pos(i), 0)).collect(),
        axioms: vec![],
    }
}

/// Agent `i` passing one unit of charge to agent `j` at any of `m`
/// locations, for every battery level pair (capacity `b`).
fn charges(i: usize, j: usize, m: u32, b: u32) -> Vec<Operator> {
    let (pos, bat) = (|i: usize| 2 * i, |i: usize| 2 * i + 1);
    let mut operators = Vec::new();
    for l in 0..m {
        for x in 1..=b {
            for y in 0..b {
                operators.push(Operator {
                    name: format!("charge-{i}-{j}-{l}-{x}-{y}"),
                    pre: vec![
                        Fact::new(pos(i), l),
                        Fact::new(pos(j), l),
                        Fact::new(bat(i), x),
                        Fact::new(bat(j), y),
                    ],
                    eff: vec![
                        Effect::new(Fact::new(bat(i), x - 1)),
                        Effect::new(Fact::new(bat(j), y + 1)),
                    ],
                    cost: 1,
                });
            }
        }
    }

    operators
}

/// The generalized strategy finds the paper's factoring: one leaf per
/// agent with its battery, an empty center, the charge actions global.
#[test]
fn every_agent_is_a_leaf() {
    let task = agents(3, 3, 8);
    let factoring = Factoring::generalized(&task);

    assert!(factoring.center().is_empty(), "{factoring:?}");
    assert_eq!(factoring.leaves(), &[vec![0, 1], vec![2, 3], vec![4, 5]]);
    let score = factoring.score(&task).expect("searchable");
    assert_eq!(score.mobile_leaves, 3);
}

/// Decoupled search over the generalized factoring finds the optimal cost
/// under every configuration of the cross-checks and stores fewer states
/// than explicit search; no star strategy has two mobile leaves here.
#[test]
fn agents_are_solved_optimally_with_fewer_states() {
    let task = agents(3, 3, 8);
    let generalized = Factoring::generalized(&task);
    let expected = optimal_cost(&task);
    assert!(expected.is_some());
    assert_eq!(checked_cost(&task, &generalized), expected);

    let config = Config {
        pruning: Pruning::Basic,
        ..Config::default()
    };
    let stored = |factoring: &Factoring| {
        config
            .search(&task, factoring)
            .expect("searchable")
            .statistics
            .stored
    };
    let explicit = Factoring::new(&task, vec![]).expect("valid");
    assert!(stored(&generalized) < stored(&explicit));

    for strategy in Strategy::ALL
        .into_iter()
        .filter(|&s| s != Strategy::Generalized)
    {
        let score = Factoring::with_strategy(&task, strategy)
            .score(&task)
            .expect("searchable");
        assert!(score.mobile_leaves < 2, "{strategy:?}: {score:?}");
    }
}

/// `Factoring::auto` picks the generalized factoring here: it is the only
/// candidate with more than one mobile leaf.
#[test]
fn auto_picks_the_generalized_factoring() {
    let task = agents(3, 3, 8);

    assert_eq!(Factoring::auto(&task), Factoring::generalized(&task));
}

/// On random tasks, the generalized factoring often has two or more mobile
/// leaves without being a star; there, decoupled search over it matches
/// Dijkstra under every cross-check configuration.
#[test]
fn non_star_generalized_factorings_match_dijkstra() {
    let params = RandomTaskParams {
        num_vars: 5,
        max_domain: 3,
        num_ops: 10,
        ..RandomTaskParams::default()
    };
    let mut non_star = 0;
    for seed in 0..100 {
        let task = random_task(seed, params);
        let factoring = Factoring::generalized(&task);
        let Ok(score) = factoring.score(&task) else {
            continue;
        };
        if score.mobile_leaves < 2 || factoring.check_star(&task).is_ok() {
            continue;
        }

        non_star += 1;
        assert_eq!(
            checked_cost(&task, &factoring),
            optimal_cost(&task),
            "seed {seed}"
        );
    }

    assert!(
        non_star > 10,
        "only {non_star} non-star generalized factorings"
    );
}
