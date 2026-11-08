//! The renderer's submission/failure state (spec 012). A small ownership boundary, not a command
//! scheduler: it decides when pending GPU resources may be released and when the renderer has entered
//! an unrecoverable terminal state after a fence wait could not confirm completion.

/// Where a unit of GPU work stands relative to its fence.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum SubmitState {
    /// Nothing submitted yet — ordinary rollback may free the staged resources immediately.
    NotSubmitted,
    /// Submitted; the GPU may still read the staging/destination/command resources. Must not free.
    Pending,
    /// A successful wait confirmed completion — the resources are safe to free.
    Completed,
    /// A wait could not confirm completion (and the device is not confirmed lost). Unrecoverable:
    /// leak rather than free possibly-in-use memory; never reuse.
    Terminal,
}

/// The outcome of a submit/wait step driving a [`SubmitState`] transition.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum SubmitStep {
    Submitted,
    WaitOk,
    WaitFailed,
}

impl SubmitState {
    pub(crate) fn advance(self, step: SubmitStep) -> SubmitState {
        match (self, step) {
            (SubmitState::Terminal, _) => SubmitState::Terminal,
            (SubmitState::NotSubmitted, SubmitStep::Submitted) => SubmitState::Pending,
            (SubmitState::Pending, SubmitStep::WaitOk) => SubmitState::Completed,
            (SubmitState::Pending, SubmitStep::WaitFailed) => SubmitState::Terminal,
            // Any other combination is out-of-sequence misuse; hold the current state defensively.
            (state, _) => state,
        }
    }

    /// Pending GPU resources may be released only when nothing was submitted or completion is
    /// confirmed — never while `Pending` or after `Terminal`.
    pub(crate) fn may_release_pending(self) -> bool {
        matches!(self, SubmitState::NotSubmitted | SubmitState::Completed)
    }

    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, SubmitState::Terminal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_failure_after_submit_is_terminal_and_blocks_release() {
        let s = SubmitState::NotSubmitted;
        assert!(s.may_release_pending()); // before submit: ordinary rollback is fine
        let pending = s.advance(SubmitStep::Submitted);
        assert_eq!(pending, SubmitState::Pending);
        assert!(!pending.may_release_pending()); // pending: must not free in-flight resources

        let terminal = pending.advance(SubmitStep::WaitFailed);
        assert_eq!(terminal, SubmitState::Terminal);
        assert!(terminal.is_terminal());
        assert!(!terminal.may_release_pending()); // terminal: leak, never free
        assert_eq!(terminal.advance(SubmitStep::WaitOk), SubmitState::Terminal); // stays terminal

        let completed = pending.advance(SubmitStep::WaitOk);
        assert_eq!(completed, SubmitState::Completed);
        assert!(completed.may_release_pending()); // confirmed complete: safe to free
    }
}
