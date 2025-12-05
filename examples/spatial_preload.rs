//! Spatial preloading example - archetype_asset's UNIQUE EDGE!
//!
//! This demonstrates position-based asset prediction and preloading.

use archetype_asset::{DistancePredictor, SpatialPredictor};
use glam::Vec3;

fn main() {
    println!("archetype_asset Spatial Preloading Demo");
    println!("======================================\n");
    println!("This is archetype_asset's UNIQUE FEATURE!\n");

    // Create a distance-based predictor with 100 unit radius
    let predictor = DistancePredictor::new(100.0);

    // Simulate player position and velocity
    let position = Vec3::new(100.0, 0.0, 50.0);
    let velocity = Some(Vec3::new(10.0, 0.0, 5.0)); // Moving forward-right

    println!("Player position: {:?}", position);
    println!("Player velocity: {:?}", velocity);
    println!("Prediction radius: {} units", predictor.prediction_radius());

    // Predict which assets will be needed
    let predicted_assets = predictor.predict_assets(position, velocity);

    if predicted_assets.is_empty() {
        println!("\nNo assets predicted (implement scene graph for real predictions)");
    } else {
        println!("\nPredicted assets to preload:");
        for (i, asset) in predicted_assets.iter().enumerate() {
            println!("  {}. {}", i + 1, asset);
        }
    }

    println!("\n=== How Spatial Preloading Works ===");
    println!("1. Track player position and velocity");
    println!("2. Query scene graph for nearby assets");
    println!("3. Predict movement direction");
    println!("4. Preload assets in predicted path");
    println!("5. Result: Zero loading hitches!");

    println!("\nSpatial preload example complete!");
}
