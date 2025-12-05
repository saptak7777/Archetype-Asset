//! Benchmark: Cache hit performance

use archetype_asset::{AssetCache, MockGpu};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn cache_hit_perf_benchmark(c: &mut Criterion) {
    let gpu = MockGpu::new();
    let cache = AssetCache::new(gpu, 100 * 1024 * 1024);

    // Benchmark cache hit check performance
    c.bench_function("cache_hit_check", |b| {
        b.iter(|| {
            // Simulate cache hit rate calculation
            let metrics = cache.metrics();
            black_box(metrics.cache_hit_rate())
        })
    });

    c.bench_function("cache_memory_usage", |b| {
        b.iter(|| black_box(cache.memory_usage()))
    });
}

criterion_group!(benches, cache_hit_perf_benchmark);
criterion_main!(benches);
