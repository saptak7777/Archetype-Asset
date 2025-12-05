//! Async streaming example for archetype_asset

#[cfg(feature = "runtime-tokio")]
use archetype_asset::{MockGpu, AssetManager, AsyncModelHandle};

#[cfg(feature = "runtime-tokio")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use std::time::Duration;
    
    println!("archetype_asset Async Streaming Demo");
    println!("====================================\n");
    
    let gpu = MockGpu::new();
    let manager = archetype_asset::AssetManager::new(gpu);
    
    // Start async loading (uncomment when you have a model file)
    // let handle = manager.load_model_async("assets/model.glb").await?;
    
    // Poll for progress
    // while handle.is_loading() {
    //     println!("Loading: {:.0}%", handle.progress() * 100.0);
    //     tokio::time::sleep(Duration::from_millis(100)).await;
    // }
    
    // if let Some(model) = handle.get_model() {
    //     println!("Loaded {} meshes!", model.meshes.len());
    // }
    
    println!("Async streaming example complete!");
    Ok(())
}

#[cfg(not(feature = "runtime-tokio"))]
fn main() {
    println!("This example requires the runtime-tokio feature.");
    println!("Run with: cargo run --example async_streaming --features runtime-tokio");
}
