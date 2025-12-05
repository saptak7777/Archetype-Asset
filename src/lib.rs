//! archetype_asset - Fast, modular asset system with spatial preloading
//!
//! # Features
//! - GPU abstraction (Vulkan, DirectX, Metal via trait)
//! - Async runtime abstraction (Tokio, async-std, custom)
//! - Generic asset caching with monomorphization
//! - Spatial preloading (unique!)
//! - LOD simplification
//! - Zero-copy streaming
//!
//! # Quick Start
//!
//! ```ignore
//! use archetype_asset::{AssetCache, MockGpu};
//!
//! let gpu = MockGpu::new();
//! let cache = AssetCache::new(gpu, 100 * 1024 * 1024);
//! let model = cache.get_or_load_model("model.glb")?;
//! ```
//!
//! # Feature Flags
//!
//! - `gpu-vulkan`: Enable Vulkan GPU backend
//! - `runtime-tokio`: Enable Tokio async runtime
//! - `lod`: Enable mesh simplification with meshopt
//! - `spatial-preload`: Enable spatial preloading features

// Core modules
pub mod cache;
pub mod gpu;
pub mod loader;
pub mod lod;
pub mod runtime;
pub mod spatial;

// Support modules
pub mod async_loading;
pub mod model;
pub mod renderer;
pub mod texture;

// Error types
mod error;
pub use error::{AssetError, Result};

// Re-export main types from cache
pub use cache::metrics::{AssetMetrics, AssetMetricsHandle};
pub use cache::pool::{AssetMemoryPool, MeshData, TextureData, VertexData};
pub use cache::AssetCache;

// Re-export GPU types
pub use gpu::mock::MockGpu;
#[cfg(feature = "gpu-vulkan")]
pub use gpu::vulkan::VulkanDevice;
pub use gpu::{BufferUsage, GpuDevice, GpuError, GpuResult, GpuTextureFormat, TextureDescriptor};

// Re-export runtime types
pub use runtime::mock::MockSpawner;
#[cfg(feature = "runtime-tokio")]
pub use runtime::tokio_impl::TokioSpawner;
pub use runtime::{AsyncSpawner, JoinHandle};

// Re-export model types
pub use model::{
    AlphaMode, LoadedModel, Material, Mesh, ModelError, ModelHandle, ModelLoader, Node,
    PrimitiveType, Transform,
};

// Re-export texture types
pub use texture::{Texture, TextureError, TextureFormat, TextureLoader};

// Re-export LOD types
#[cfg(feature = "lod")]
pub use lod::MeshoptSimplifier;
pub use lod::{
    default_thresholds, generate_lod_levels, DefaultSimplifier, LodModel, MeshSimplifier,
};

// Re-export async loading types
pub use async_loading::{AsyncAssetError, AsyncModelHandle, LoadState};

// Re-export spatial types
pub use spatial::{DistancePredictor, SpatialPredictor};

// Re-export renderer types
pub use renderer::Vertex;

// Version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_mock_gpu_available() {
        let _gpu = MockGpu::new();
    }
}
