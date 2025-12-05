//! Benchmark: Memory footprint

use archetype_asset::{AssetMemoryPool, MeshData, VertexData};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn memory_footprint_benchmark(c: &mut Criterion) {
    c.bench_function("memory_pool_creation", |b| {
        b.iter(|| {
            let pool = AssetMemoryPool::new();
            black_box(pool)
        })
    });

    c.bench_function("mesh_data_alloc", |b| {
        let pool = AssetMemoryPool::new();
        b.iter(|| {
            let mesh_data = MeshData {
                vertices: vec![0.0; 1000],
                indices: vec![0; 500],
                material_index: None,
            };
            let alloc = pool.alloc_mesh_data(mesh_data);
            black_box(alloc)
        })
    });

    c.bench_function("vertex_data_alloc", |b| {
        let pool = AssetMemoryPool::new();
        b.iter(|| {
            let vertex_data = VertexData {
                positions: vec![[0.0, 0.0, 0.0]; 100],
                normals: vec![[0.0, 1.0, 0.0]; 100],
                tex_coords: vec![[0.0, 0.0]; 100],
            };
            let alloc = pool.alloc_vertex_data(vertex_data);
            black_box(alloc)
        })
    });
}

criterion_group!(benches, memory_footprint_benchmark);
criterion_main!(benches);
