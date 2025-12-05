//! Async runtime abstraction for flexible async execution
//!
//! This module provides traits and implementations for async runtime operations,
//! allowing the asset system to work with any async runtime (tokio, async-std, etc.)

pub mod mock;
#[cfg(feature = "runtime-tokio")]
pub mod tokio_impl;

use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;

/// A boxed future that can be sent across threads
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Handle to a spawned async task
///
/// This is a type-erased handle that allows checking task completion.
#[derive(Debug)]
pub struct JoinHandle {
    inner: Box<dyn std::any::Any + Send>,
}

impl JoinHandle {
    /// Create a new join handle
    pub fn new<T: Send + 'static>(handle: T) -> Self {
        Self {
            inner: Box::new(handle),
        }
    }

    /// Try to downcast to a specific handle type
    pub fn downcast<T: 'static>(self) -> Option<T> {
        self.inner.downcast::<T>().ok().map(|b| *b)
    }
}

/// Async task spawner trait
///
/// This trait abstracts async task spawning, allowing the asset system
/// to work with any async runtime.
///
/// # Example
/// ```ignore
/// let spawner = TokioSpawner::new();
/// spawner.spawn(async {
///     // Async work here
/// });
/// ```
pub trait AsyncSpawner: Send + Sync + Clone + Debug {
    /// Spawn an async task
    ///
    /// The task will run in the background and can be awaited via the returned handle.
    fn spawn<F>(&self, task: F) -> JoinHandle
    where
        F: Future<Output = ()> + Send + 'static;

    /// Spawn a task that returns a value
    fn spawn_with_result<F, T>(&self, task: F) -> JoinHandle
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static;

    /// Get the name of this runtime (for debugging)
    fn runtime_name(&self) -> &'static str;

    /// Block on a future (if supported by the runtime)
    ///
    /// Returns None if blocking is not supported.
    fn block_on<F, T>(&self, _future: F) -> Option<T>
    where
        F: Future<Output = T>,
    {
        None // Default: blocking not supported
    }
}

// Re-export implementations
pub use mock::MockSpawner;

#[cfg(feature = "runtime-tokio")]
pub use tokio_impl::TokioSpawner;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_handle_downcast() {
        let handle = JoinHandle::new(42u32);
        let value = handle.downcast::<u32>();
        assert_eq!(value, Some(42));
    }

    #[test]
    fn test_join_handle_wrong_type() {
        let handle = JoinHandle::new(42u32);
        let value = handle.downcast::<String>();
        assert!(value.is_none());
    }
}
