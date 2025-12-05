//! Benchmark: GLTF loading performance
//!
//! This benchmark measures the time to load and parse GLTF/GLB files.

use archetype_asset::{AssetCache, MockGpu, ModelLoader};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Benchmark cache operations without actual file I/O
fn cache_operations_benchmark(c: &mut Criterion) {
    let gpu = MockGpu::new();
    let cache = AssetCache::new(gpu.clone(), 100 * 1024 * 1024);

    c.bench_function("cache_creation", |b| {
        b.iter(|| {
            let cache = AssetCache::new(gpu.clone(), 100 * 1024 * 1024);
            black_box(cache)
        })
    });

    c.bench_function("cache_memory_check", |b| {
        b.iter(|| black_box(cache.memory_usage()))
    });
}

/// Benchmark model loader creation
fn loader_creation_benchmark(c: &mut Criterion) {
    c.bench_function("model_loader_new", |b| {
        b.iter(|| {
            let loader = ModelLoader::new();
            black_box(loader)
        })
    });
}

/// Benchmark varying cache sizes
fn cache_size_benchmark(c: &mut Criterion) {
    let gpu = MockGpu::new();
    let sizes = [1024, 1024 * 1024, 100 * 1024 * 1024];

    let mut group = c.benchmark_group("cache_sizes");
    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let cache = AssetCache::new(gpu.clone(), size);
                black_box(cache.memory_usage())
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    cache_operations_benchmark,
    loader_creation_benchmark,
    cache_size_benchmark
);
criterion_main!(benches);
