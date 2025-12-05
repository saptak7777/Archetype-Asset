//! Integration tests for async loading pipeline

use archetype_asset::{AsyncModelHandle, LoadState, ModelHandle};

#[test]
fn test_async_model_handle_lifecycle() {
    let handle = AsyncModelHandle::new(ModelHandle::new());

    // Initial state should be pending
    assert!(handle.is_loading());
    assert!(!handle.is_ready());
    assert!(!handle.is_failed());

    // Progress should be 0
    assert_eq!(handle.progress(), 0.0);
}

#[test]
fn test_load_state_variants() {
    // Test all LoadState variants
    let pending = LoadState::Pending;
    let loading = LoadState::LoadingBasic;
    let streaming = LoadState::StreamingTextures(0.5);
    let finalizing = LoadState::FinalizingGeometry;
    let failed = LoadState::Failed("test error".to_string());

    assert_eq!(pending, LoadState::Pending);
    assert_ne!(pending, loading);

    if let LoadState::StreamingTextures(p) = streaming {
        assert_eq!(p, 0.5);
    }

    if let LoadState::Failed(msg) = failed {
        assert!(msg.contains("test"));
    }
}
