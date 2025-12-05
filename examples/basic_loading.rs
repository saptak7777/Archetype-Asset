//! Basic loading example for archetype_asset

use archetype_asset::{AssetCache, MockGpu};

fn main() -> anyhow::Result<()> {
    // Create a mock GPU device (no real hardware needed)
    let gpu = MockGpu::new();

    // Create asset cache with 100MB limit
    let cache = AssetCache::new(gpu, 100 * 1024 * 1024);

    println!("archetype_asset v{}", archetype_asset::VERSION);
    println!("Cache memory usage: {} bytes", cache.memory_usage());

    // Load a model (if you have one)
    // let model = cache.get_or_load_model("assets/model.glb")?;
    // println!("Loaded {} meshes", model.meshes.len());

    println!("Basic loading example complete!");
    Ok(())
}
