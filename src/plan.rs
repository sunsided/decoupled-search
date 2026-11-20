//! Sequential plans for the original task and their validation.

use alloc::vec::Vec;
use core::fmt;

use sas_plus::{OpId, Task, TaskError};

/// A sequential plan for the original task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Operators in execution order, applicable from the initial state.
    pub ops: Vec<OpId>,
    /// Sum of the operator costs.
    pub cost: u64,
}

/// Why [`Plan::validate`] rejected a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanError {
    /// The task failed [`Task::validate`], so plans for it cannot be checked.
    InvalidTask(TaskError),
    /// Step `step` names operator `op`, which the task does not have.
    UnknownOperator {
        /// Index of the offending step in [`Plan::ops`].
        step: usize,
        /// The operator id it names.
        op: OpId,
    },
    /// The preconditions of operator `op` do not hold before step `step`.
    NotApplicable {
        /// Index of the offending step in [`Plan::ops`].
        step: usize,
        /// The operator applied at that step.
        op: OpId,
    },
    /// Every step applies, but the goal does not hold at the end.
    GoalNotReached,
    /// The plan is valid, but its operators cost `actual`, not the
    /// [`Plan::cost`] it claims.
    CostMismatch {
        /// The cost stored in [`Plan::cost`].
        claimed: u64,
        /// The sum of the operator costs.
        actual: u64,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTask(e) => write!(f, "invalid task: {e}"),
            Self::UnknownOperator { step, op } => {
                write!(f, "step {step} names unknown operator {op}")
            }
            Self::NotApplicable { step, op } => {
                write!(f, "operator {op} at step {step} is not applicable")
            }
            Self::GoalNotReached => write!(f, "the plan does not reach the goal"),
            Self::CostMismatch { claimed, actual } => {
                write!(f, "the plan claims cost {claimed} but costs {actual}")
            }
        }
    }
}

impl core::error::Error for PlanError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::InvalidTask(e) => Some(e),
            _ => None,
        }
    }
}

impl Plan {
    /// Checks the plan against `task` with explicit-state semantics,
    /// independent of decoupled search: starting from the initial state
    /// with derived variables evaluated, every operator must be applicable,
    /// conditional effects fire on the state before the operator, derived
    /// variables are re-evaluated after it, the goal must hold at the end,
    /// and [`Plan::cost`] must be the sum of the operator costs.
    ///
    /// # Errors
    ///
    /// - [`PlanError::InvalidTask`] if `task` fails [`Task::validate`].
    /// - [`PlanError::UnknownOperator`] for the first step naming an
    ///   operator the task lacks.
    /// - [`PlanError::NotApplicable`] for the first step whose operator is
    ///   not applicable.
    /// - [`PlanError::GoalNotReached`] if the final state misses the goal.
    /// - [`PlanError::CostMismatch`] if the plan is valid but
    ///   [`Plan::cost`] is wrong.
    ///
    /// # Examples
    ///
    /// ```
    /// use decoupled_search::{Plan, PlanError};
    /// use sas_plus::{Effect, Fact, Operator, State, Task, Variable};
    ///
    /// let task = Task {
    ///     variables: vec![Variable::new("x", 2)],
    ///     operators: vec![Operator {
    ///         name: "set".into(),
    ///         pre: vec![Fact::new(0, 0)],
    ///         eff: vec![Effect::new(Fact::new(0, 1))],
    ///         cost: 3,
    ///     }],
    ///     initial: State(vec![0]),
    ///     goal: vec![Fact::new(0, 1)],
    ///     axioms: vec![],
    /// };
    ///
    /// assert_eq!(Plan { ops: vec![0], cost: 3 }.validate(&task), Ok(()));
    /// assert_eq!(
    ///     Plan { ops: vec![0, 0], cost: 6 }.validate(&task),
    ///     Err(PlanError::NotApplicable { step: 1, op: 0 })
    /// );
    /// assert_eq!(
    ///     Plan { ops: vec![], cost: 0 }.validate(&task),
    ///     Err(PlanError::GoalNotReached)
    /// );
    /// ```
    pub fn validate(&self, task: &Task) -> Result<(), PlanError> {
        task.validate().map_err(PlanError::InvalidTask)?;

        let mut state = task.initial_state();
        let mut actual = 0_u64;
        for (step, &op) in self.ops.iter().enumerate() {
            let Some(operator) = task.operators.get(op) else {
                return Err(PlanError::UnknownOperator { step, op });
            };

            if !task.is_applicable(op, &state) {
                return Err(PlanError::NotApplicable { step, op });
            }

            state = task.apply(op, &state);
            actual = actual.saturating_add(u64::from(operator.cost));
        }

        if !task.is_goal(&state) {
            return Err(PlanError::GoalNotReached);
        }

        if actual != self.cost {
            return Err(PlanError::CostMismatch {
                claimed: self.cost,
                actual,
            });
        }

        Ok(())
    }
}
