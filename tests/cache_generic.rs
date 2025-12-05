//! Integration tests for generic AssetCache

use archetype_asset::{AssetCache, MockGpu};

#[test]
fn test_cache_generic_over_gpu() {
    let gpu = MockGpu::new();
    let cache = AssetCache::new(gpu, 100 * 1024 * 1024); // 100MB

    assert_eq!(cache.memory_usage(), 0);
}

#[test]
fn test_cache_clear() {
    let gpu = MockGpu::new();
    let cache = AssetCache::new(gpu, 50 * 1024 * 1024);

    cache.clear();
    assert_eq!(cache.memory_usage(), 0);
}

#[test]
fn test_cache_metrics() {
    let gpu = MockGpu::new();
    let cache = AssetCache::new(gpu, 100 * 1024 * 1024);

    // Access metrics
    let metrics = cache.metrics();
    assert_eq!(metrics.cache_hit_rate(), 0.0);
}
