use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Tracks performance metrics for asset loading and caching
#[derive(Debug, Default)]
pub struct AssetMetrics {
    load_times: RwLock<HashMap<String, Duration>>,
    load_counts: RwLock<HashMap<String, u64>>,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    total_memory: AtomicU64,
}

impl AssetMetrics {
    /// Create a new instance of AssetMetrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the load time for an asset
    pub fn record_load_time(&self, path: String, duration: Duration) {
        let mut load_times = self.load_times.write();
        load_times.insert(path, duration);
    }

    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record memory usage for an asset
    pub fn record_memory_usage(&self, path: String, bytes: usize) {
        self.total_memory.fetch_add(bytes as u64, Ordering::Relaxed);
        let mut counts = self.load_counts.write();
        *counts.entry(path).or_insert(0) += 1;
    }

    /// Get the cache hit rate as a percentage
    pub fn cache_hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f32;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f32;

        if hits + misses > 0.0 {
            hits / (hits + misses) * 100.0
        } else {
            0.0
        }
    }

    /// Get the total memory used by loaded assets in bytes
    pub fn total_memory_usage(&self) -> u64 {
        self.total_memory.load(Ordering::Relaxed)
    }

    /// Get the average load time for an asset
    pub fn average_load_time(&self, path: &str) -> Option<Duration> {
        self.load_times.read().get(path).cloned()
    }

    /// Get the load count for an asset
    pub fn load_count(&self, path: &str) -> u64 {
        *self.load_counts.read().get(path).unwrap_or(&0)
    }

    /// Get all recorded load times
    pub fn all_load_times(&self) -> HashMap<String, Duration> {
        self.load_times.read().clone()
    }
}

/// A thread-safe wrapper around AssetMetrics
#[derive(Debug, Clone, Default)]
pub struct AssetMetricsHandle(Arc<AssetMetrics>);

impl AssetMetricsHandle {
    /// Create a new metrics handle
    pub fn new() -> Self {
        Self(Arc::new(AssetMetrics::new()))
    }

    /// Get a reference to the underlying metrics
    pub fn inner(&self) -> &AssetMetrics {
        &self.0
    }
}

impl std::ops::Deref for AssetMetricsHandle {
    type Target = AssetMetrics;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
