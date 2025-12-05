//! Mock async spawner for testing
//!
//! Provides a mock async spawner that can run tasks synchronously
//! or drop them entirely for testing purposes.

use super::{AsyncSpawner, JoinHandle};
use std::future::Future;

/// Spawn behavior for MockSpawner
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockSpawnBehavior {
    /// Drop tasks immediately (don't execute)
    Drop,
    /// Block on tasks synchronously using a simple executor
    BlockSync,
}

/// Mock async spawner for testing
///
/// This spawner can either drop tasks immediately or run them
/// synchronously for testing purposes.
#[derive(Clone, Debug)]
pub struct MockSpawner {
    behavior: MockSpawnBehavior,
}

impl Default for MockSpawner {
    fn default() -> Self {
        Self::new()
    }
}

impl MockSpawner {
    /// Create a new mock spawner that drops tasks
    pub fn new() -> Self {
        Self {
            behavior: MockSpawnBehavior::Drop,
        }
    }

    /// Create a mock spawner with specific behavior
    pub fn with_behavior(behavior: MockSpawnBehavior) -> Self {
        Self { behavior }
    }

    /// Create a mock spawner that runs tasks synchronously
    pub fn blocking() -> Self {
        Self {
            behavior: MockSpawnBehavior::BlockSync,
        }
    }
}

impl AsyncSpawner for MockSpawner {
    fn spawn<F>(&self, task: F) -> JoinHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        match self.behavior {
            MockSpawnBehavior::Drop => {
                // Just drop the task
                drop(task);
                JoinHandle::new(())
            }
            MockSpawnBehavior::BlockSync => {
                // Run synchronously using a simple block_on
                futures::executor::block_on(task);
                JoinHandle::new(())
            }
        }
    }

    fn spawn_with_result<F, T>(&self, task: F) -> JoinHandle
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        match self.behavior {
            MockSpawnBehavior::Drop => {
                drop(task);
                JoinHandle::new(None::<T>)
            }
            MockSpawnBehavior::BlockSync => {
                let result = futures::executor::block_on(task);
                JoinHandle::new(Some(result))
            }
        }
    }

    fn runtime_name(&self) -> &'static str {
        "Mock"
    }

    fn block_on<F, T>(&self, future: F) -> Option<T>
    where
        F: Future<Output = T>,
    {
        match self.behavior {
            MockSpawnBehavior::Drop => None,
            MockSpawnBehavior::BlockSync => Some(futures::executor::block_on(future)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_spawner_drop() {
        let spawner = MockSpawner::new();
        let handle = spawner.spawn(async {
            // This should be dropped
            panic!("Should not run");
        });
        // Just check we got a handle back
        let _ = handle;
    }

    #[test]
    fn test_mock_spawner_blocking() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let spawner = MockSpawner::blocking();
        let ran = Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        spawner.spawn(async move {
            ran_clone.store(true, Ordering::SeqCst);
        });

        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn test_mock_spawner_with_result() {
        let spawner = MockSpawner::blocking();
        let handle = spawner.spawn_with_result(async { 42u32 });
        
        let result = handle.downcast::<Option<u32>>();
        assert_eq!(result, Some(Some(42)));
    }

    #[test]
    fn test_mock_spawner_block_on() {
        let spawner = MockSpawner::blocking();
        let result = spawner.block_on(async { 42u32 });
        assert_eq!(result, Some(42));
    }
}
