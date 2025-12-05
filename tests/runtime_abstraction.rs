//! Integration tests for async runtime abstraction

use archetype_asset::{AsyncSpawner, MockSpawner};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[test]
fn test_mock_spawner_integration() {
    let spawner = MockSpawner::blocking();

    let executed = Arc::new(AtomicBool::new(false));
    let executed_clone = Arc::clone(&executed);

    spawner.spawn(async move {
        executed_clone.store(true, Ordering::SeqCst);
    });

    // In blocking mode, should execute immediately
    assert!(executed.load(Ordering::SeqCst));
}

#[test]
fn test_spawner_trait_bound() {
    fn spawn_task<S: AsyncSpawner>(spawner: &S) {
        spawner.spawn(async {});
    }

    let spawner = MockSpawner::new();
    spawn_task(&spawner);
}
