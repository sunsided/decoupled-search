//! Random tasks with a star-shaped causal graph by construction, and the
//! automatic star factorings on them.

use decoupled_search::{Error, Factoring, Strategy};
use sas_plus::random::{RandomTaskParams, Rng, random_task};
use sas_plus::{Effect, Fact, Operator, State, Task, VarId, Variable};

use crate::checked_cost;
use crate::dijkstra::optimal_cost;

/// Number of random star tasks per test.
pub const SEEDS: u64 = 200;

/// Variables of a star task: center variables first, then each leaf's.
struct Layout {
    /// The center variables.
    center: Vec<VarId>,
    /// The variables of each leaf.
    leaves: Vec<Vec<VarId>>,
}

/// A random task whose causal graph is a star: leaf operators read the
/// center and their own leaf and write their own leaf; global operators read
/// the center and at most one leaf, and write the center and possibly that
/// leaf. A third of the operators cost 0. The goal lies on one or two leaf
/// variables, none true initially.
pub fn random_star_task(seed: u64) -> Task {
    let mut rng = Rng::new(seed);
    let layout = random_layout(&mut rng);
    let variables: Vec<Variable> = (0..layout.center.len()
        + layout.leaves.iter().flatten().count())
        .map(|v| Variable::new(format!("v{v}"), pick(&mut rng, 2, 3)))
        .collect();
    let domain = |v: VarId| variables[v].domain_size;

    let mut operators = Vec::new();
    for leaf in &layout.leaves {
        for _ in 0..pick(&mut rng, 3, 5) {
            let pre = random_facts(&mut rng, &[&layout.center, leaf], 2, &domain);
            let eff = random_facts(&mut rng, &[leaf], 2, &domain);
            operators.push(operator(&mut rng, pre, eff));
        }
    }

    for _ in 0..pick(&mut rng, 4, 7) {
        let leaf = &layout.leaves[pick_index(&mut rng, layout.leaves.len())];
        let pre = random_facts(&mut rng, &[&layout.center, leaf], 2, &domain);
        let mut eff = random_facts(&mut rng, &[&layout.center], 1, &domain);
        if rng.range(0, 1) == 1 {
            eff.extend(random_facts(&mut rng, &[leaf], 1, &domain));
        }

        operators.push(operator(&mut rng, pre, eff));
    }

    let initial = State(
        (0..variables.len())
            .map(|v| pick(&mut rng, 0, domain(v) - 1))
            .collect(),
    );
    let leaf_vars: Vec<VarId> = layout.leaves.iter().flatten().copied().collect();
    let goal = random_facts(&mut rng, &[&leaf_vars], 2, &domain)
        .into_iter()
        .map(|f| Fact::new(f.var, (initial.0[f.var] + 1) % domain(f.var)))
        .collect();

    Task {
        variables,
        operators,
        initial,
        goal,
        axioms: vec![],
    }
}

/// One or two center variables and two or three leaves of one or two
/// variables each.
fn random_layout(rng: &mut Rng) -> Layout {
    let center: Vec<VarId> = (0..=pick_index(rng, 2)).collect();
    let mut next = center.len();
    let leaves = (0..pick(rng, 2, 3))
        .map(|_| {
            let size = pick_index(rng, 2) + 1;
            let leaf = (next..next + size).collect();
            next += size;
            leaf
        })
        .collect();

    Layout { center, leaves }
}

/// An operator with the given preconditions and effects and a random cost
/// in `0..=3`, zero for a third of the draws.
fn operator(rng: &mut Rng, pre: Vec<Fact>, eff: Vec<Fact>) -> Operator {
    let cost = if rng.range(0, 2) == 0 {
        0
    } else {
        pick(rng, 1, 3)
    };
    Operator {
        name: "op".into(),
        pre,
        eff: eff.into_iter().map(Effect::new).collect(),
        cost,
    }
}

/// Between one and `max` facts on distinct variables drawn from `groups`,
/// with random values; empty only if the groups are.
fn random_facts(
    rng: &mut Rng,
    groups: &[&Vec<VarId>],
    max: u64,
    domain: &impl Fn(VarId) -> u32,
) -> Vec<Fact> {
    let mut vars: Vec<VarId> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let count = usize::try_from(pick(rng, 1, u32::try_from(max).expect("small"))).expect("small");
    let mut facts = Vec::new();
    while facts.len() < count && !vars.is_empty() {
        let var = vars.swap_remove(pick_index(rng, vars.len()));
        facts.push(Fact::new(var, pick(rng, 0, domain(var) - 1)));
    }

    facts
}

/// A uniform value in `lo..=hi`.
fn pick(rng: &mut Rng, lo: u32, hi: u32) -> u32 {
    u32::try_from(rng.range(lo.into(), hi.into())).expect("in range")
}

/// A uniform index in `0..len`; `len` must be positive.
fn pick_index(rng: &mut Rng, len: usize) -> usize {
    usize::try_from(rng.range(0, u64::try_from(len - 1).expect("small"))).expect("small")
}

/// The automatic factoring strategies, and the automatic choice among them.
pub fn strategies(task: &Task) -> [(&'static str, Factoring); 5] {
    [
        ("fork", Factoring::fork(task)),
        ("inverted-fork", Factoring::inverted_fork(task)),
        ("incident-arcs", Factoring::incident_arcs(task)),
        ("max-leaves", Factoring::max_leaves(task)),
        ("auto", Factoring::auto(task)),
    ]
}

/// Every strategy yields a strict star on star tasks and on unstructured
/// random tasks.
#[test]
fn strategies_yield_star_factorings() {
    let random = (0..SEEDS).map(|seed| random_task(seed, RandomTaskParams::default()));
    for task in (0..SEEDS).map(random_star_task).chain(random) {
        for (name, factoring) in strategies(&task) {
            assert_eq!(factoring.check_star(&task), Ok(()), "{name}: {factoring:?}");
        }
    }
}

/// The greedy strategies find several leaves on star tasks, and decoupled
/// search under them matches Dijkstra.
#[test]
fn greedy_strategies_match_dijkstra_on_star_tasks() {
    let mut multi_leaf_runs = 0;
    for seed in 0..SEEDS {
        let task = random_star_task(seed);
        let expected = optimal_cost(&task);
        for (name, factoring) in strategies(&task) {
            assert_eq!(
                checked_cost(&task, &factoring),
                expected,
                "seed {seed}, {name}"
            );
            if factoring.leaves().len() >= 2 && expected.is_some_and(|c| c > 0) {
                multi_leaf_runs += 1;
            }
        }
    }

    assert!(
        multi_leaf_runs > SEEDS / 2,
        "only {multi_leaf_runs} multi-leaf runs"
    );
}

/// Two binary variables and one operator with the given preconditions and
/// effects.
fn two_vars(pre: Vec<Fact>, eff: Vec<Fact>) -> Task {
    Task {
        variables: vec![Variable::new("a", 2), Variable::new("b", 2)],
        operators: vec![Operator {
            name: "op".into(),
            pre,
            eff: eff.into_iter().map(Effect::new).collect(),
            cost: 1,
        }],
        initial: State(vec![0, 0]),
        goal: vec![Fact::new(1, 1)],
        axioms: vec![],
    }
}

/// Operators that read one leaf and write another, or write two leaves,
/// break the star; a center between them does not.
#[test]
fn star_check_rejects_coupled_leaves() {
    let reads_other = two_vars(vec![Fact::new(0, 1)], vec![Fact::new(1, 1)]);
    let split = Factoring::new(&reads_other, vec![vec![0], vec![1]]).expect("valid");
    assert_eq!(
        split.check_star(&reads_other),
        Err(Error::NotStar { from: 0, to: 1 })
    );

    let writes_both = two_vars(vec![], vec![Fact::new(0, 1), Fact::new(1, 1)]);
    let split = Factoring::new(&writes_both, vec![vec![0], vec![1]]).expect("valid");
    assert_eq!(
        split.check_star(&writes_both),
        Err(Error::NotStar { from: 0, to: 1 })
    );

    let one_leaf = Factoring::new(&writes_both, vec![vec![1]]).expect("valid");
    assert_eq!(one_leaf.check_star(&writes_both), Ok(()));

    let wider =
        Factoring::new(&random_task(0, RandomTaskParams::default()), vec![vec![3]]).expect("valid");
    assert_eq!(
        wider.check_star(&writes_both),
        Err(Error::FactoringMismatch)
    );
}

/// `auto` picks a strategy's factoring with at least two mobile leaves
/// whose leaf-action share is the best of all such strategies, and
/// abstains (no leaves) only if no strategy has two mobile leaves; on star
/// tasks it mostly finds two.
#[test]
fn auto_picks_the_best_scored_strategy() {
    let mut two_mobile = 0;
    for seed in 0..SEEDS {
        let task = random_star_task(seed);
        let auto = Factoring::auto(&task);
        let score = auto.score(&task).expect("searchable");
        let candidates: Vec<_> = Strategy::ALL
            .iter()
            .filter_map(|&s| {
                let factoring = Factoring::with_strategy(&task, s);
                let score = factoring.score(&task).ok()?;
                Some((factoring, score))
            })
            .filter(|(_, score)| score.mobile_leaves >= 2)
            .collect();
        if auto.leaves().is_empty() {
            assert!(candidates.is_empty(), "seed {seed}: abstained");
            continue;
        }

        assert!(score.mobile_leaves >= 2, "seed {seed}: {score:?}");
        assert!(
            candidates.iter().any(|(f, _)| *f == auto),
            "seed {seed}: {auto:?}"
        );
        for (_, other) in &candidates {
            assert!(
                other.leaf_action_share() <= score.leaf_action_share(),
                "seed {seed}: {other:?} beats {score:?}"
            );
        }

        two_mobile += 1;
    }

    assert!(
        two_mobile > SEEDS / 2,
        "only {two_mobile} with two mobile leaves"
    );
}

/// A truck `t` (0) that drives freely, packages `b` (2) and `d` (4) moved
/// by leaf actions that read `t` or `a` (1), `a` moved by the truck and
/// moving `b`, and `c` (3) moved only together with the truck.
///
/// Causal graph: `t -> a, b, c`, `c -> t`, `a -> b, d`. Incident arcs:
/// `t` 4, `a` 3, `b` 2, `c` 2, `d` 1.
fn truck_with_frozen_passenger() -> Task {
    let op = |pre: Vec<Fact>, eff: Vec<Fact>| Operator {
        name: "op".into(),
        pre,
        eff: eff.into_iter().map(Effect::new).collect(),
        cost: 1,
    };
    Task {
        variables: (0..5).map(|v| Variable::new(format!("v{v}"), 2)).collect(),
        operators: vec![
            op(vec![], vec![Fact::new(0, 1)]),
            op(vec![Fact::new(0, 0)], vec![Fact::new(1, 1)]),
            op(vec![Fact::new(0, 0)], vec![Fact::new(2, 1)]),
            op(vec![Fact::new(1, 1)], vec![Fact::new(2, 0)]),
            op(vec![], vec![Fact::new(0, 0), Fact::new(3, 1)]),
            op(vec![Fact::new(1, 1)], vec![Fact::new(4, 1)]),
        ],
        initial: State(vec![0; 5]),
        goal: vec![Fact::new(2, 1), Fact::new(3, 1), Fact::new(4, 1)],
        axioms: vec![],
    }
}

/// Incident arcs (Figure 2 of Gnad, Poser and Hoffmann 2017): center `{t}`
/// leaves `{a, b, d}` (mobile) and `{c}` (frozen), one mobile leaf; center
/// `{t, a}` leaves `{b}`, `{c}` and `{d}`, two mobile. The strategy keeps
/// the latter; taking the first candidate with two leaves would not.
#[test]
fn incident_arcs_keeps_the_most_mobile_leaves() {
    let task = truck_with_frozen_passenger();
    let factoring = Factoring::incident_arcs(&task);
    assert_eq!(factoring.center(), &[0, 1]);
    assert_eq!(factoring.leaves(), &[vec![2], vec![3], vec![4]]);
    assert_eq!(factoring.score(&task).expect("searchable").mobile_leaves, 2);
    assert_eq!(checked_cost(&task, &factoring), optimal_cost(&task));
}

/// Independent sets (Figure 1): the only maximum independent set is
/// `{b, c, d}`. Growing cannot add `t` (adjacent to `b` and `c`) or `a`
/// (adjacent to `b` and `d`); the frozen `{c}` goes back to the center;
/// growing again adds `t` to `{b}` (now its only neighbouring leaf), which
/// keeps two mobile leaves, and then `c`, whose only neighbour is `t`.
#[test]
fn max_leaves_grows_the_independent_set() {
    let task = truck_with_frozen_passenger();
    let factoring = Factoring::max_leaves(&task);
    assert_eq!(factoring.center(), &[1]);
    assert_eq!(factoring.leaves(), &[vec![0, 2, 3], vec![4]]);
    let score = factoring.score(&task).expect("searchable");
    assert_eq!((score.mobile_leaves, score.leaf_actions), (2, 5));
    assert_eq!(checked_cost(&task, &factoring), optimal_cost(&task));

    // The fork (leaves b, d) has three leaf actions and the inverted fork
    // one mobile leaf ({t, c}); five of six wins.
    assert_eq!(Factoring::auto(&task), factoring);
}

/// A task where no strategy finds a mobile leaf gets the factoring without
/// leaves.
#[test]
fn auto_falls_back_to_explicit_search() {
    // Every variable is too large for a leaf.
    let mut task = two_vars(vec![], vec![Fact::new(0, 1), Fact::new(1, 1)]);
    for v in &mut task.variables {
        v.domain_size = 1 << 17;
    }

    let auto = Factoring::auto(&task);
    assert!(auto.leaves().is_empty(), "{auto:?}");
    assert_eq!(auto.center(), &[0, 1]);
}
