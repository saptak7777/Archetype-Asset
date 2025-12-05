//! GPU abstraction layer for backend-agnostic asset loading
//!
//! This module provides traits and implementations for GPU operations,
//! allowing the asset system to work with any GPU backend.

pub mod mock;
#[cfg(feature = "gpu-vulkan")]
pub mod vulkan;

use std::fmt::Debug;
use thiserror::Error;

/// Error type for GPU operations
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Buffer allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Buffer upload failed: {0}")]
    UploadFailed(String),

    #[error("Texture creation failed: {0}")]
    TextureCreationFailed(String),

    #[error("Invalid buffer size: {0}")]
    InvalidSize(usize),

    #[error("Device lost")]
    DeviceLost,

    #[error("Out of memory")]
    OutOfMemory,
}

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

/// Buffer usage flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// Vertex buffer
    Vertex,
    /// Index buffer
    Index,
    /// Uniform buffer
    Uniform,
    /// Storage buffer
    Storage,
    /// Staging buffer for transfers
    Staging,
}

/// Texture format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTextureFormat {
    /// RGBA 8-bit with sRGB color space
    Rgba8Srgb,
    /// RGBA 8-bit unorm
    Rgba8Unorm,
    /// RGB 8-bit with sRGB color space (no alpha)
    Rgb8Srgb,
    /// Single channel 8-bit
    R8Unorm,
    /// Depth 32-bit float
    Depth32Float,
    /// Depth 24 + Stencil 8
    Depth24Stencil8,
}

/// Texture descriptor for creation
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Texture format
    pub format: GpuTextureFormat,
    /// Number of mip levels (1 = no mipmaps)
    pub mip_levels: u32,
    /// Generate mipmaps automatically
    pub generate_mipmaps: bool,
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            format: GpuTextureFormat::Rgba8Srgb,
            mip_levels: 1,
            generate_mipmaps: false,
        }
    }
}

/// Core GPU device trait for backend-agnostic operations
///
/// This trait abstracts GPU operations, allowing the asset system to work
/// with any GPU backend (Vulkan, DX12, Metal, Mock, etc.) through associated types.
///
/// # Associated Types
/// - `Buffer`: The buffer type for this GPU backend
/// - `Texture`: The texture type for this GPU backend
///
/// # Example
/// ```ignore
/// // Using with a mock GPU for testing
/// let gpu = MockGpu::new();
/// let buffer = gpu.allocate_buffer(1024, BufferUsage::Vertex)?;
/// gpu.upload_buffer_data(&buffer, 0, &vertex_data)?;
/// ```
pub trait GpuDevice: Send + Sync + Clone + Debug {
    /// Buffer type for this GPU backend
    type Buffer: Clone + Send + Sync + Debug;

    /// Texture type for this GPU backend
    type Texture: Clone + Send + Sync + Debug;

    /// Allocate a GPU buffer
    ///
    /// # Arguments
    /// * `size` - Size in bytes
    /// * `usage` - How the buffer will be used
    ///
    /// # Returns
    /// A handle to the allocated buffer
    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> GpuResult<Self::Buffer>;

    /// Upload data to a buffer
    ///
    /// # Arguments
    /// * `buffer` - Target buffer
    /// * `offset` - Byte offset into the buffer
    /// * `data` - Data to upload
    fn upload_buffer_data(
        &self,
        buffer: &Self::Buffer,
        offset: usize,
        data: &[u8],
    ) -> GpuResult<()>;

    /// Create a texture from data
    ///
    /// # Arguments
    /// * `desc` - Texture descriptor
    /// * `data` - Pixel data (can be empty for render targets)
    fn create_texture(&self, desc: &TextureDescriptor, data: &[u8]) -> GpuResult<Self::Texture>;

    /// Destroy a buffer (optional cleanup)
    ///
    /// Most backends handle cleanup through Drop, but this allows explicit cleanup.
    fn destroy_buffer(&self, _buffer: Self::Buffer) {
        // Default: let Drop handle it
    }

    /// Destroy a texture (optional cleanup)
    fn destroy_texture(&self, _texture: Self::Texture) {
        // Default: let Drop handle it
    }

    /// Get the name of this GPU backend (for debugging)
    fn backend_name(&self) -> &'static str;
}

// Re-export implementations
pub use mock::MockGpu;

#[cfg(feature = "gpu-vulkan")]
pub use vulkan::VulkanDevice;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_usage_debug() {
        let usage = BufferUsage::Vertex;
        assert_eq!(format!("{:?}", usage), "Vertex");
    }

    #[test]
    fn test_texture_descriptor_default() {
        let desc = TextureDescriptor::default();
        assert_eq!(desc.width, 1);
        assert_eq!(desc.height, 1);
        assert_eq!(desc.mip_levels, 1);
    }
}
