//! Error types for archetype_asset

use thiserror::Error;

/// Main error type for asset operations
#[derive(Error, Debug)]
pub enum AssetError {
    #[error("Model error: {0}")]
    Model(#[from] crate::model::ModelError),

    #[error("Texture error: {0}")]
    Texture(#[from] crate::texture::TextureError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("GLTF error: {0}")]
    Gltf(#[from] gltf::Error),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("GPU error: {0}")]
    Gpu(#[from] crate::gpu::GpuError),

    #[error("Cache error: {0}")]
    Cache(String),
}

/// Result type alias for asset operations
pub type Result<T> = std::result::Result<T, AssetError>;
