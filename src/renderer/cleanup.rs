// Scoped rollback for Vulkan handles while their owning wrapper is being constructed.
pub(super) struct Cleanup<F: FnOnce()> {
    action: Option<F>,
}

impl<F: FnOnce()> Cleanup<F> {
    pub(super) fn new(action: F) -> Self {
        Self {
            action: Some(action),
        }
    }

    pub(super) fn disarm(mut self) {
        self.action.take();
    }
}

impl<F: FnOnce()> Drop for Cleanup<F> {
    fn drop(&mut self) {
        if let Some(action) = self.action.take() {
            action();
        }
    }
}

/// Build a collection atomically, releasing earlier items if a later creation fails.
pub(super) fn collect_resources<I, T, E>(
    items: impl IntoIterator<Item = I>,
    mut create: impl FnMut(I) -> Result<T, E>,
    mut destroy: impl FnMut(T),
) -> Result<Vec<T>, E> {
    let mut resources = Vec::new();
    for item in items {
        match create(item) {
            Ok(resource) => resources.push(resource),
            Err(error) => {
                for resource in resources.into_iter().rev() {
                    destroy(resource);
                }
                return Err(error);
            }
        }
    }
    Ok(resources)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn collection_rolls_back_at_every_failure_point() {
        for failure in 0..4 {
            let mut destroyed = Vec::new();
            let result = collect_resources(
                0..4,
                |item| if item == failure { Err(item) } else { Ok(item) },
                |item| destroyed.push(item),
            );
            assert_eq!(result, Err(failure));
            assert_eq!(destroyed, (0..failure).rev().collect::<Vec<_>>());
        }
    }

    #[test]
    fn collection_transfers_all_resources_on_success() {
        let resources = collect_resources(0..4, Ok::<_, ()>, |_| panic!("ownership transferred"));
        assert_eq!(resources, Ok(vec![0, 1, 2, 3]));
        let empty = collect_resources(0..0, Ok::<_, ()>, |_| panic!("nothing allocated"));
        assert_eq!(empty, Ok(vec![]));
    }

    #[test]
    fn rollback_runs_once_on_early_error() {
        let calls = Cell::new(0);
        let result: Result<(), ()> = (|| {
            let _cleanup = Cleanup::new(|| calls.set(calls.get() + 1));
            Err(())?;
            Ok(())
        })();
        assert!(result.is_err());
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn successful_transfer_disarms_cleanup() {
        let calls = Cell::new(0);
        Cleanup::new(|| calls.set(calls.get() + 1)).disarm();
        assert_eq!(calls.get(), 0);
    }
}
