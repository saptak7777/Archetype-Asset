//! GLTF/GLB model loading
//!
//! This module provides functions to load GLTF and GLB files.
//! The actual implementation is in `model.rs`, this module provides
//! a convenience API.

use crate::model::{LoadedModel, ModelError, ModelLoader};
use std::path::Path;

/// Load a GLB file from a path
pub fn load_glb_file<P: AsRef<Path>>(path: P) -> Result<LoadedModel, ModelError> {
    let data = std::fs::read(path)?;
    load_glb_bytes(&data)
}

/// Load a GLB from bytes
pub fn load_glb_bytes(data: &[u8]) -> Result<LoadedModel, ModelError> {
    let loader = ModelLoader::new();
    loader.load_glb(data)
}

/// Load a GLTF file from a path (with external buffers/textures)
pub fn load_gltf_file<P: AsRef<Path>>(path: P) -> Result<LoadedModel, ModelError> {
    // For now, only GLB (embedded) is fully supported
    // GLTF with external files would need additional work
    let data = std::fs::read(path)?;

    // Try to load as GLB first
    load_glb_bytes(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_glb_bytes_empty() {
        let result = load_glb_bytes(&[]);
        assert!(result.is_err());
    }
}
