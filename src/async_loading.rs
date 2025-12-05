//! Async model loading with streaming support
//!
//! This module provides async loading capabilities for models,
//! with progress tracking and state management.

use crate::{LoadedModel, ModelHandle};
use parking_lot::RwLock as SyncRwLock;
use std::sync::Arc;
use thiserror::Error;

/// Error type for async asset loading operations
#[derive(Error, Debug)]
pub enum AsyncAssetError {
    #[error("Failed to load model: {0}")]
    ModelLoadError(#[from] anyhow::Error),

    #[error("Asset loading was cancelled")]
    LoadingCancelled,

    #[error("Asset loading failed: {0}")]
    LoadingFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Represents the current state of an async model load
#[derive(Debug, Clone, PartialEq)]
pub enum LoadState {
    /// Loading has not started yet
    Pending,

    /// Currently loading basic model data
    LoadingBasic,

    /// Currently streaming textures with progress (0.0 to 1.0)
    StreamingTextures(f32),

    /// Finalizing geometry after textures are loaded
    FinalizingGeometry,

    /// Loading completed successfully
    Completed(LoadedModel),

    /// Loading failed with an error message
    Failed(String),
}

/// Handle to an asynchronously loading model
pub struct AsyncModelHandle {
    pub handle: ModelHandle,
    state: Arc<SyncRwLock<LoadState>>,
}

impl AsyncModelHandle {
    /// Create a new async model handle
    pub fn new(handle: ModelHandle) -> Self {
        Self {
            handle,
            state: Arc::new(SyncRwLock::new(LoadState::Pending)),
        }
    }

    /// Get the current load state
    pub fn state(&self) -> LoadState {
        self.state.read().clone()
    }

    /// Check if loading is complete
    pub fn is_ready(&self) -> bool {
        matches!(*self.state.read(), LoadState::Completed(_))
    }

    /// Check if loading failed
    pub fn is_failed(&self) -> bool {
        matches!(*self.state.read(), LoadState::Failed(_))
    }

    /// Check if loading is still in progress
    pub fn is_loading(&self) -> bool {
        matches!(
            *self.state.read(),
            LoadState::Pending
                | LoadState::LoadingBasic
                | LoadState::StreamingTextures(_)
                | LoadState::FinalizingGeometry
        )
    }

    /// Get the loading progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        match &*self.state.read() {
            LoadState::Pending => 0.0,
            LoadState::LoadingBasic => 0.1,
            LoadState::StreamingTextures(p) => 0.1 + (p * 0.7),
            LoadState::FinalizingGeometry => 0.9,
            LoadState::Completed(_) => 1.0,
            LoadState::Failed(_) => 0.0,
        }
    }

    /// Get the loaded model if loading is complete
    pub fn get_model(&self) -> Option<LoadedModel> {
        match &*self.state.read() {
            LoadState::Completed(model) => Some(model.clone()),
            _ => None,
        }
    }

    /// Get the state arc for internal use
    #[allow(dead_code)]
    pub fn state_arc(&self) -> Arc<SyncRwLock<LoadState>> {
        Arc::clone(&self.state)
    }

    /// Set the state (for async loading orchestration)
    #[allow(dead_code)]
    pub fn set_state(&self, state: LoadState) {
        *self.state.write() = state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_state_eq() {
        assert_eq!(LoadState::Pending, LoadState::Pending);
        assert_ne!(LoadState::Pending, LoadState::LoadingBasic);
    }

    #[test]
    fn test_async_model_handle_progress() {
        let handle = AsyncModelHandle::new(ModelHandle::new());

        assert_eq!(handle.progress(), 0.0);
        assert!(handle.is_loading());
        assert!(!handle.is_ready());
        assert!(!handle.is_failed());
    }

    #[test]
    fn test_async_model_handle_state_transitions() {
        let handle = AsyncModelHandle::new(ModelHandle::new());

        handle.set_state(LoadState::LoadingBasic);
        assert!(matches!(handle.state(), LoadState::LoadingBasic));

        handle.set_state(LoadState::StreamingTextures(0.5));
        assert!(matches!(handle.state(), LoadState::StreamingTextures(_)));

        handle.set_state(LoadState::FinalizingGeometry);
        assert!(matches!(handle.state(), LoadState::FinalizingGeometry));
    }
}
