# archetype_asset

**Fast, modular asset system with spatial preloading for Rust game engines.**

[![Crates.io](https://img.shields.io/crates/v/archetype_asset.svg)](https://crates.io/crates/archetype_asset)
[![Documentation](https://docs.rs/archetype_asset/badge.svg)](https://docs.rs/archetype_asset)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Features

- **🎮 GPU Abstraction** - Generic over GPU backends (Vulkan, Mock for testing)
- **⚡ Async Runtime Abstraction** - Works with Tokio, async-std, or custom runtimes
- **📦 Smart Caching** - LRU cache with memory management and metrics
- **🗺️ Spatial Preloading** - *Unique!* Position-based asset prediction
- **📐 LOD System** - Level-of-detail mesh simplification
- **🎬 GLTF/GLB Loading** - Full PBR material and scene hierarchy support

## Quick Start

```rust
use archetype_asset::{AssetCache, MockGpu};
use archetype_asset::loader::gltf::load_glb_file;

// Create GPU and cache
let gpu = MockGpu::new();
let cache = AssetCache::new(gpu, 100 * 1024 * 1024); // 100MB cache

// Load a GLB model directly
let model = load_glb_file("model.glb")?;
println!("Loaded {} meshes", model.meshes.len());

// Or use the cache for automatic caching (recommended)
let cached_model = cache.get_or_load_model("model.glb")?;
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `gpu-mock` | Mock GPU for testing | ✅ |
| `gpu-vulkan` | Vulkan GPU backend | ❌ |
| `runtime-mock` | Mock async runtime | ✅ |
| `runtime-tokio` | Tokio async runtime | ❌ |
| `lod` | LOD mesh simplification | ❌ |
| `spatial-preload` | Spatial preloading | ❌ |
| `metrics` | Performance metrics | ❌ |

### Enable Vulkan + Tokio

```toml
[dependencies]
archetype_asset = { version = "0.1.2", features = ["gpu-vulkan", "runtime-tokio"] }
```

## API Overview

### GPU Abstraction

```rust
use archetype_asset::{GpuDevice, MockGpu, BufferUsage};

// Any GPU backend implements GpuDevice
let gpu = MockGpu::new();
let buffer = gpu.allocate_buffer(1024, BufferUsage::Vertex)?;
gpu.upload_buffer_data(&buffer, 0, &data)?;
```

### Async Runtime Abstraction

```rust
use archetype_asset::{AsyncSpawner, MockSpawner};

let spawner = MockSpawner::blocking();
spawner.spawn(async {
    // Async work here
});
```

### Asset Cache

```rust
use archetype_asset::{AssetCache, MockGpu};

let cache = AssetCache::new(MockGpu::new(), 100 * 1024 * 1024);

// Load with caching (returns Arc<LoadedModel>)
let model = cache.get_or_load_model("model.glb")?;

// Check memory usage
println!("Cache using {} bytes", cache.memory_usage());

// Performance metrics
let hit_rate = cache.metrics().cache_hit_rate();
```

### Model Loading

```rust
use archetype_asset::{ModelLoader, Vertex};

// Load from bytes
let loader = ModelLoader::new();
let model = loader.load_glb(&glb_bytes)?;

// Access mesh data
for mesh in &model.meshes {
    let vertices = mesh.vertices();
    let vertex_count = vertices.vertices.len() / Vertex::floats_per_vertex();
    println!("Mesh has {} vertices", vertex_count);
}
```

### Spatial Preloading (Unique!)

```rust
use archetype_asset::{SpatialPredictor, DistancePredictor};
use glam::Vec3;

// Predict assets based on player position
let predictor = DistancePredictor::new(100.0); // 100 unit radius
let needed_assets = predictor.predict_assets(player_position, Some(velocity));
```

### LOD System

```rust
use archetype_asset::{LodModel, DefaultSimplifier, generate_lod_levels, default_thresholds};

// Generate 4 LOD levels
let simplifier = DefaultSimplifier::new();
let lod_levels = generate_lod_levels(&mesh, 4, &simplifier);
let thresholds = default_thresholds(4);

let lod_model = LodModel::new(lod_levels, thresholds);

// Select LOD based on screen size
let mesh_to_render = lod_model.get_lod(screen_size);
```

### Texture Loading

```rust
use archetype_asset::TextureLoader;

// Synchronous loading
let loader = TextureLoader::new();
let texture = loader.load_file_sync("texture.png")?;
println!("Loaded {}x{} texture", texture.width, texture.height);
```

## Performance

Benchmarks on typical hardware:

| Operation | Time |
|-----------|------|
| Cache hit check | 5.4 ns |
| Cache memory check | 0.5 ns |
| Cache creation | 291 ns |
| Mesh data allocation | 138 ns |
| Memory pool creation | 1.5 µs |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Author

**Saptak Santra**
