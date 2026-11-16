//! Shared ownership and validation mechanisms for renderer-owned resources (meshes, textures).
//!
//! Resources stored in the renderer (e.g. meshes in `mesh.rs`, textures in `texture.rs`) use a
//! two-part handle: an `owner_id` that identifies which renderer instance created them, and a
//! `slot` that indexes the store within that renderer. This module provides:
//!
//! - `next_owner_id()`: A monotonic counter ensuring each renderer instance gets a unique owner ID
//!   (starting at 1, so 0 is a safe sentinel for "invalid/default").
//! - `resolve_handle()`: Validates that a `(owner_id, slot)` pair points to a valid resource in a
//!   specific renderer's store.

use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide monotonic owner ID counter. Starts at 0; `next_owner_id()` pre-increments so the
/// first real ID is 1, and zero is never a valid owner ID.
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Return the next unique owner ID for a renderer instance. Calls are monotonically increasing and
/// never repeat. The first call returns 1 (never 0), so a default/zeroed ID is always invalid.
pub(crate) fn next_owner_id() -> u64 {
    allocate_owner_id(&COUNTER)
}

fn allocate_owner_id(counter: &AtomicU64) -> u64 {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .expect("renderer owner identifiers exhausted")
        + 1
}

/// Validate a `(owner_id, slot)` resource handle against a specific renderer's owner ID and store
/// length.
///
/// Returns `Some(slot)` if the owner ID matches and the slot is within bounds; `None` otherwise.
/// This is the single shared implementation of handle validation, called by mesh and texture stores
/// when resolving handles.
///
/// # Arguments
/// - `owner`: The owner ID from the handle being resolved.
/// - `slot`: The resource index from the handle being resolved.
/// - `my_owner`: The owner ID of the renderer store being checked against.
/// - `store_len`: The number of valid slots in the renderer's store.
pub(crate) fn resolve_handle(
    owner: u64,
    slot: usize,
    my_owner: u64,
    store_len: usize,
) -> Option<usize> {
    if owner == my_owner && slot < store_len {
        Some(slot)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_ids_fail_closed_on_exhaustion_without_wrapping() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(allocate_owner_id(&counter), u64::MAX);
        assert!(std::panic::catch_unwind(|| allocate_owner_id(&counter)).is_err());
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn next_owner_id_never_repeats_across_many_calls() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            assert!(seen.insert(next_owner_id()), "owner id repeated");
        }
    }

    #[test]
    fn resolve_handle_matching_owner_and_valid_slot() {
        let owner = next_owner_id();
        assert_eq!(resolve_handle(owner, 0, owner, 10), Some(0));
        assert_eq!(resolve_handle(owner, 5, owner, 10), Some(5));
        assert_eq!(resolve_handle(owner, 9, owner, 10), Some(9));
    }

    #[test]
    fn resolve_handle_mismatched_owner_returns_none() {
        let owner1 = next_owner_id();
        let owner2 = next_owner_id();
        assert_ne!(owner1, owner2);
        // Even with a valid slot, mismatched owner should fail
        assert_eq!(resolve_handle(owner1, 0, owner2, 10), None);
    }

    #[test]
    fn resolve_handle_out_of_bounds_slot_returns_none() {
        let owner = next_owner_id();
        // In-bounds slot succeeds
        assert_eq!(resolve_handle(owner, 9, owner, 10), Some(9));
        // Out-of-bounds slots fail
        assert_eq!(resolve_handle(owner, 10, owner, 10), None);
        assert_eq!(resolve_handle(owner, 100, owner, 10), None);
    }

    #[test]
    fn resolve_handle_zero_owner_never_matches() {
        let owner = next_owner_id();
        // Even with a valid slot, owner ID 0 should never match
        assert_eq!(resolve_handle(0, 0, owner, 10), None);
    }
}
