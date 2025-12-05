//! Asset caching with LRU eviction policy
//!
//! This module provides a caching layer for loaded assets with
//! memory management and spatial preloading support.

pub mod metrics;
pub mod pool;

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use xxhash_rust::xxh3::Xxh3;

#[cfg(feature = "runtime-tokio")]
use tokio::sync::Mutex as TokioMutex;

use crate::gpu::GpuDevice;
use crate::model::{LoadedModel, Material, ModelLoader, Node};
#[cfg(feature = "runtime-tokio")]
use crate::spatial::preloader::AssetPreloader;
use crate::texture::Texture;
use metrics::AssetMetricsHandle;

// Re-export pool types
pub use pool::{AssetMemoryPool, MeshData, TextureData, VertexData};

/// Represents a cached asset with metadata
struct CachedAsset<T> {
    asset: Arc<T>,
    size: usize,
    #[allow(dead_code)]
    last_accessed: u64,
}

/// Manages cached assets with LRU eviction policy
///
/// Generic over the GPU device type for maximum flexibility.
pub struct AssetCache<G: GpuDevice = crate::gpu::mock::MockGpu> {
    models: Arc<RwLock<HashMap<u64, CachedAsset<LoadedModel>>>>,
    textures: Arc<RwLock<HashMap<u64, CachedAsset<Texture>>>>,
    model_lru: Arc<RwLock<VecDeque<u64>>>,
    texture_lru: Arc<RwLock<VecDeque<u64>>>,
    max_memory: usize,
    current_memory: Arc<AtomicUsize>,
    gpu: G,
    #[cfg(feature = "runtime-tokio")]
    preloader: Arc<TokioMutex<AssetPreloader>>,
    /// Performance metrics for asset loading and caching
    metrics: AssetMetricsHandle,
}

impl<G: GpuDevice> Clone for AssetCache<G> {
    fn clone(&self) -> Self {
        // Create a new cache with the same configuration but empty state
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            textures: Arc::new(RwLock::new(HashMap::new())),
            model_lru: Arc::new(RwLock::new(VecDeque::new())),
            texture_lru: Arc::new(RwLock::new(VecDeque::new())),
            max_memory: self.max_memory,
            current_memory: Arc::new(AtomicUsize::new(0)),
            gpu: self.gpu.clone(),
            #[cfg(feature = "runtime-tokio")]
            preloader: self.preloader.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

impl<G: GpuDevice> AssetCache<G> {
    /// Creates a new AssetCache with the specified GPU device and memory limit
    pub fn new(gpu: G, max_memory: usize) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            textures: Arc::new(RwLock::new(HashMap::new())),
            model_lru: Arc::new(RwLock::new(VecDeque::new())),
            texture_lru: Arc::new(RwLock::new(VecDeque::new())),
            max_memory,
            current_memory: Arc::new(AtomicUsize::new(0)),
            gpu,
            #[cfg(feature = "runtime-tokio")]
            preloader: Arc::new(TokioMutex::new(AssetPreloader::new(100.0))),
            metrics: AssetMetricsHandle::new(),
        }
    }

    /// Get a reference to the GPU device
    pub fn gpu(&self) -> &G {
        &self.gpu
    }

    /// Gets a model from cache or loads it if not present
    pub fn get_or_load_model<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Arc<LoadedModel>, anyhow::Error> {
        let path = path.as_ref();
        let path_hash = self.hash_path(path);
        let current_time = Self::current_timestamp();

        // Check cache first
        {
            let models = self.models.read().unwrap();
            if let Some(cached) = models.get(&path_hash) {
                // Record cache hit
                self.metrics.record_cache_hit();

                // Update last accessed time
                let mut model_lru = self.model_lru.write().unwrap();
                if let Some(pos) = model_lru.iter().position(|&id| id == path_hash) {
                    model_lru.remove(pos);
                }
                model_lru.push_back(path_hash);
                return Ok(Arc::clone(&cached.asset));
            }

            // Record cache miss
            self.metrics.record_cache_miss();
        }

        // Load the model with timing
        let start_time = std::time::Instant::now();
        let model = self.load_model_direct(path)?;
        let size = self.estimate_model_size(&model);

        // Record load time and memory usage
        let load_duration = start_time.elapsed();
        let path_str = path.to_string_lossy().to_string();
        self.metrics
            .record_load_time(path_str.clone(), load_duration);
        self.metrics.record_memory_usage(path_str, size);

        // Check if we need to evict old assets
        self.evict_if_needed(size);

        // Insert into cache
        let model_arc = Arc::new(model);
        let cached = CachedAsset {
            asset: Arc::clone(&model_arc),
            size,
            last_accessed: current_time,
        };

        {
            let mut models = self.models.write().unwrap();
            models.insert(path_hash, cached);
            self.model_lru.write().unwrap().push_back(path_hash);
        }

        self.current_memory.fetch_add(size, Ordering::SeqCst);

        Ok(model_arc)
    }

    /// Clears all cached assets
    pub fn clear(&self) {
        {
            self.models.write().unwrap().clear();
            self.textures.write().unwrap().clear();
            self.model_lru.write().unwrap().clear();
            self.texture_lru.write().unwrap().clear();
        }
        self.current_memory.store(0, Ordering::SeqCst);
    }

    /// Gets the current memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.current_memory.load(Ordering::Relaxed)
    }

    /// Get a reference to the metrics handle
    pub fn metrics(&self) -> &AssetMetricsHandle {
        &self.metrics
    }

    /// Load a model directly without caching
    fn load_model_direct<P: AsRef<Path>>(&self, path: P) -> Result<LoadedModel, anyhow::Error> {
        let model_loader = ModelLoader::new();

        // Read the model file
        let data = std::fs::read(path)?;

        // Load the model using the GLB loader
        model_loader.load_glb(&data).map_err(Into::into)
    }

    fn estimate_model_size(&self, model: &LoadedModel) -> usize {
        let mut size = 0;

        // Estimate size of meshes
        for mesh in &model.meshes {
            // Get read access to the mesh data
            let mesh_data = mesh.vertices();

            // Calculate size based on vertex and index data
            size += mesh_data.vertices.len() * std::mem::size_of::<f32>();
            size += mesh_data.indices.len() * std::mem::size_of::<u32>();
        }

        // Add size for materials and other model data
        size += model.materials.len() * std::mem::size_of::<Material>();
        size += model.nodes.len() * std::mem::size_of::<Node>();

        size
    }

    fn evict_if_needed(&self, required_size: usize) {
        let mut current_memory = self.current_memory.load(Ordering::SeqCst);
        let max_memory = self.max_memory;

        if current_memory + required_size <= max_memory {
            return; // No need to evict
        }

        // First, try to evict models
        self.evict_from_cache::<LoadedModel>(
            &self.models,
            &self.model_lru,
            &mut current_memory,
            max_memory,
            required_size,
        );

        // If still not enough space, try to evict textures
        if current_memory + required_size > max_memory {
            self.evict_from_cache::<Texture>(
                &self.textures,
                &self.texture_lru,
                &mut current_memory,
                max_memory,
                required_size,
            );
        }

        self.current_memory.store(current_memory, Ordering::SeqCst);
    }

    fn evict_from_cache<T: 'static>(
        &self,
        cache: &RwLock<HashMap<u64, CachedAsset<T>>>,
        lru: &RwLock<VecDeque<u64>>,
        current_memory: &mut usize,
        max_memory: usize,
        required_size: usize,
    ) {
        let mut cache = cache.write().unwrap();
        let mut lru = lru.write().unwrap();

        while *current_memory + required_size > max_memory && !lru.is_empty() {
            if let Some(oldest_id) = lru.pop_front() {
                if let Some(removed) = cache.remove(&oldest_id) {
                    *current_memory -= removed.size;
                }
            }
        }
    }

    fn hash_path<P: AsRef<Path>>(&self, path: P) -> u64 {
        let path = path.as_ref();
        let mut hasher = Xxh3::new();
        path.hash(&mut hasher);

        // Include file modification time in the hash to detect changes
        if let Ok(metadata) = std::fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                    duration.as_secs().hash(&mut hasher);
                    duration.subsec_nanos().hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

// Async methods (require tokio feature)
#[cfg(feature = "runtime-tokio")]
impl<G: GpuDevice> AssetCache<G> {
    /// Gets a texture from cache or loads it if not present
    pub async fn get_or_load_texture<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<Arc<Texture>, anyhow::Error> {
        let path = path.as_ref();
        let path_hash = self.hash_path(path);
        let current_time = Self::current_timestamp();

        // Check cache first
        {
            let textures = self.textures.read().unwrap();
            if let Some(cached) = textures.get(&path_hash) {
                // Record cache hit
                self.metrics.record_cache_hit();

                // Update last accessed time
                let mut texture_lru = self.texture_lru.write().unwrap();
                if let Some(pos) = texture_lru.iter().position(|&id| id == path_hash) {
                    texture_lru.remove(pos);
                }
                texture_lru.push_back(path_hash);
                return Ok(Arc::clone(&cached.asset));
            }

            // Record cache miss
            self.metrics.record_cache_miss();
        }

        // Load the texture with timing
        let start_time = std::time::Instant::now();
        let texture = self.load_texture_direct(path).await?;
        let size = texture.data.len();

        // Record load time and memory usage
        let load_duration = start_time.elapsed();
        let path_str = path.to_string_lossy().to_string();
        self.metrics
            .record_load_time(path_str.clone(), load_duration);
        self.metrics.record_memory_usage(path_str, size);

        // Check if we need to evict old assets
        self.evict_if_needed(size);

        // Add to cache
        let texture_arc = Arc::new(texture);
        let cached = CachedAsset {
            asset: Arc::clone(&texture_arc),
            size,
            last_accessed: current_time,
        };

        // Update cache
        {
            let mut textures = self.textures.write().unwrap();
            let mut texture_lru = self.texture_lru.write().unwrap();

            textures.insert(path_hash, cached);
            texture_lru.push_back(path_hash);
            self.current_memory.fetch_add(size, Ordering::SeqCst);
        }

        Ok(texture_arc)
    }

    /// Get a reference to the asset preloader
    pub fn preloader(&self) -> &Arc<TokioMutex<AssetPreloader>> {
        &self.preloader
    }

    /// Load a texture directly without caching
    async fn load_texture_direct<P: AsRef<Path>>(&self, path: P) -> Result<Texture, anyhow::Error> {
        let texture_loader = TextureLoader::new();

        // Await the async load_file method
        let texture = texture_loader.load_file(path).await?;
        Ok(texture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpu;

    #[test]
    fn test_cache_creation() {
        let gpu = MockGpu::new();
        let cache = AssetCache::new(gpu, 100 * 1024 * 1024);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_cache_clear() {
        let gpu = MockGpu::new();
        let cache = AssetCache::new(gpu, 100 * 1024 * 1024);
        cache.clear();
        assert_eq!(cache.memory_usage(), 0);
    }
}
