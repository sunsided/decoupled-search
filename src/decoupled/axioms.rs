//! Evaluation of derived variables on parts of a state.
//!
//! A derived variable is evaluated wherever the factor that owns it lives:
//! one that depends only on center variables is a center variable,
//! recomputed after every center change; one that depends on the center and
//! a single leaf is attached to that leaf and evaluated per leaf state under
//! the current center. [`Axioms::evaluate`] runs the rules of one such
//! [`Scope`] with the semantics of [`sas_plus::Task::evaluate_axioms`].

use alloc::vec;
use alloc::vec::Vec;

use sas_plus::{Fact, Task, Value, VarId};

/// One axiom rule `conditions -> head`.
#[derive(Clone, Debug)]
struct Rule {
    /// Facts on task variables that must hold for the rule to fire.
    conditions: Vec<Fact>,
    /// The derived fact the rule sets.
    head: Fact,
}

/// The derived variables of one factor and their rules, by layer.
///
/// A scope is closed under dependencies within its factor: every derived
/// variable a rule of the scope reads is either in the scope or evaluated
/// before it (a center variable when the scope belongs to a leaf).
#[derive(Clone, Debug, Default)]
pub(crate) struct Scope {
    /// The derived task variables the scope evaluates.
    pub(crate) vars: Vec<VarId>,
    /// Rule indices per axiom layer, layers ascending, each in task order.
    layers: Vec<Vec<usize>>,
}

impl Scope {
    /// Whether the scope evaluates nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
}

/// The axioms of a task, compiled for evaluation on scratch assignments.
#[derive(Clone, Debug)]
pub(crate) struct Axioms {
    /// Every rule of the task, in task order.
    rules: Vec<Rule>,
    /// The axiom layer of every task variable, `None` for state variables.
    layer: Vec<Option<u32>>,
    /// The default value of every task variable (its value in
    /// [`Task::initial`]); derived variables are reset to it.
    defaults: Vec<Value>,
    /// The task variable of every center slot.
    center_vars: Vec<VarId>,
}

impl Axioms {
    /// Compiles the axioms of `task` for a center over `center_vars`.
    pub(crate) fn new(task: &Task, center_vars: Vec<VarId>) -> Self {
        Self {
            rules: task
                .axioms
                .iter()
                .map(|a| Rule {
                    conditions: a.conditions.clone(),
                    head: a.head,
                })
                .collect(),
            layer: task.variables.iter().map(|v| v.axiom_layer).collect(),
            defaults: task.initial.0.clone(),
            center_vars,
        }
    }

    /// A scratch assignment to all task variables: the defaults, overwritten
    /// with the center state `center` (by center slot).
    pub(crate) fn scratch(&self, center: &[Value]) -> Vec<Value> {
        let mut values = self.defaults.clone();
        for (&var, &value) in self.center_vars.iter().zip(center) {
            values[var] = value;
        }

        values
    }

    /// Re-evaluates the derived center variables of `scope` in `center` (by
    /// center slot) after its state variables changed.
    pub(crate) fn evaluate_center(&self, scope: &Scope, center: &mut [Value]) {
        if scope.is_empty() {
            return;
        }

        let mut values = self.scratch(center);
        self.evaluate(scope, &mut values);
        for (slot, &var) in self.center_vars.iter().enumerate() {
            center[slot] = values[var];
        }
    }

    /// The scope evaluating the derived variables `vars`.
    pub(crate) fn scope(&self, vars: Vec<VarId>) -> Scope {
        let mut layers: Vec<(u32, Vec<usize>)> = Vec::new();
        for (index, rule) in self.rules.iter().enumerate() {
            if !vars.contains(&rule.head.var) {
                continue;
            }

            let layer = self.layer[rule.head.var].expect("axiom heads are derived");
            match layers.iter_mut().find(|(l, _)| *l == layer) {
                Some((_, rules)) => rules.push(index),
                None => layers.push((layer, vec![index])),
            }
        }

        layers.sort_by_key(|(layer, _)| *layer);
        Scope {
            vars,
            layers: layers.into_iter().map(|(_, rules)| rules).collect(),
        }
    }

    /// The task variables the rules for `var` read directly.
    pub(crate) fn dependencies(&self, var: VarId) -> impl Iterator<Item = VarId> + '_ {
        self.rules
            .iter()
            .filter(move |r| r.head.var == var)
            .flat_map(|r| r.conditions.iter().map(|c| c.var))
    }

    /// Evaluates the variables of `scope` in `values` (indexed by task
    /// variable): resets them to their defaults, then fires the rules layer
    /// by layer until nothing changes. A variable keeps the first
    /// non-default value derived for it, as in
    /// [`sas_plus::Task::evaluate_axioms`].
    ///
    /// Only the variables the scope's rules read need meaningful values.
    pub(crate) fn evaluate(&self, scope: &Scope, values: &mut [Value]) {
        for &var in &scope.vars {
            values[var] = self.defaults[var];
        }

        for layer in &scope.layers {
            let mut changed = true;
            while changed {
                changed = false;
                for rule in layer.iter().map(|&i| &self.rules[i]) {
                    let default = self.defaults[rule.head.var];
                    if values[rule.head.var] == default
                        && rule.head.value != default
                        && rule.conditions.iter().all(|c| values[c.var] == c.value)
                    {
                        values[rule.head.var] = rule.head.value;
                        changed = true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use sas_plus::State;
    use sas_plus::random::{RandomTaskParams, Rng, random_task};

    use super::*;

    /// Evaluating all derived variables in one scope matches
    /// `Task::evaluate_axioms` on random states of random tasks.
    #[test]
    fn evaluation_matches_sas_plus() {
        let params = RandomTaskParams {
            num_derived: 3,
            ..RandomTaskParams::default()
        };
        for seed in 0..200 {
            let task = random_task(seed, params);
            let axioms = Axioms::new(&task, Vec::new());
            let derived = (0..task.num_variables())
                .filter(|&v| task.variables[v].is_derived())
                .collect();
            let scope = axioms.scope(derived);
            let mut rng = Rng::new(seed);
            for _ in 0..10 {
                let values: Vec<Value> = task
                    .variables
                    .iter()
                    .map(|v| {
                        u32::try_from(rng.range(0, u64::from(v.domain_size - 1))).expect("small")
                    })
                    .collect();
                let mut expected = State(values.clone());
                task.evaluate_axioms(&mut expected);
                let mut actual = values;
                axioms.evaluate(&scope, &mut actual);
                assert_eq!(actual, expected.0, "seed {seed}");
            }
        }
    }
}
