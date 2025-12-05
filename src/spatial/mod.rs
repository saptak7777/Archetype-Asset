//! Spatial preloading module
//!
//! Provides position-based asset prediction and preloading.
//! This is archetype_asset's UNIQUE EDGE!

pub mod preloader;

// Re-export main types
pub use preloader::{AssetPreloader, SceneGraph, SceneGraphMock};

use glam::Vec3;

/// Trait for predicting which assets will be needed based on spatial position
///
/// This is the core abstraction for spatial preloading.
pub trait SpatialPredictor: Send + Sync {
    /// Predict assets that will be needed at the given position
    fn predict_assets(&self, position: Vec3, velocity: Option<Vec3>) -> Vec<String>;

    /// Get the prediction radius
    fn prediction_radius(&self) -> f32;

    /// Update internal state
    fn update(&mut self);
}

/// Default spatial predictor using distance-based heuristics
#[derive(Debug, Clone)]
pub struct DistancePredictor {
    radius: f32,
}

impl Default for DistancePredictor {
    fn default() -> Self {
        Self::new(100.0)
    }
}

impl DistancePredictor {
    /// Create a new distance-based predictor
    pub fn new(radius: f32) -> Self {
        Self { radius }
    }
}

impl SpatialPredictor for DistancePredictor {
    fn predict_assets(&self, _position: Vec3, _velocity: Option<Vec3>) -> Vec<String> {
        Vec::new()
    }

    fn prediction_radius(&self) -> f32 {
        self.radius
    }

    fn update(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_predictor_creation() {
        let predictor = DistancePredictor::new(50.0);
        assert_eq!(predictor.prediction_radius(), 50.0);
    }

    #[test]
    fn test_distance_predictor_default() {
        let predictor = DistancePredictor::default();
        assert_eq!(predictor.prediction_radius(), 100.0);
    }
}
