//! Tokio async runtime implementation
//!
//! Provides integration with the Tokio async runtime.

use super::{AsyncSpawner, JoinHandle};
use std::future::Future;

/// Tokio-based async spawner
///
/// Spawns tasks on the Tokio runtime.
#[derive(Clone, Debug, Default, Copy)]
pub struct TokioSpawner;

impl TokioSpawner {
    /// Create a new Tokio spawner
    pub fn new() -> Self {
        Self
    }
}

impl AsyncSpawner for TokioSpawner {
    fn spawn<F>(&self, task: F) -> JoinHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let handle = tokio::spawn(task);
        JoinHandle::new(handle)
    }

    fn spawn_with_result<F, T>(&self, task: F) -> JoinHandle
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let handle = tokio::spawn(task);
        JoinHandle::new(handle)
    }

    fn runtime_name(&self) -> &'static str {
        "Tokio"
    }

    fn block_on<F, T>(&self, future: F) -> Option<T>
    where
        F: Future<Output = T>,
    {
        // Try to get the current runtime handle
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // If we're already in a tokio context, use block_in_place
            Some(tokio::task::block_in_place(|| handle.block_on(future)))
        } else {
            // Create a new runtime for blocking
            let rt = tokio::runtime::Runtime::new().ok()?;
            Some(rt.block_on(future))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_tokio_spawner() {
        let spawner = TokioSpawner::new();
        let ran = Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        let handle = spawner.spawn(async move {
            ran_clone.store(true, Ordering::SeqCst);
        });

        // Give the task time to run
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        
        assert!(ran.load(Ordering::SeqCst));
        let _ = handle;
    }

    #[tokio::test]
    async fn test_tokio_spawner_with_result() {
        let spawner = TokioSpawner::new();
        let handle = spawner.spawn_with_result(async { 42u32 });
        
        // The handle contains a JoinHandle<u32>
        let inner = handle.downcast::<tokio::task::JoinHandle<u32>>();
        assert!(inner.is_some());
        
        if let Some(inner) = inner {
            let result = inner.await.unwrap();
            assert_eq!(result, 42);
        }
    }

    #[test]
    fn test_tokio_runtime_name() {
        let spawner = TokioSpawner::new();
        assert_eq!(spawner.runtime_name(), "Tokio");
    }
}
