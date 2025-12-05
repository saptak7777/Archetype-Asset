//! Asset preloading system with spatial awareness.
//!
//! This module provides a system for asynchronously preloading assets
//! based on the player's position and predicted movement patterns.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use glam::Vec3;
use parking_lot::Mutex;

/// Trait for scene graph functionality needed by the asset preloader
pub trait SceneGraph: Send + Sync {
    /// Find assets near the given position within the specified radius
    fn find_assets_near(&self, position: Vec3, radius: f32) -> Vec<String>;
}

/// A simple scene graph mock for testing
pub struct SceneGraphMock;

impl SceneGraph for SceneGraphMock {
    fn find_assets_near(&self, _position: Vec3, _radius: f32) -> Vec<String> {
        vec![]
    }
}

/// Handles background preloading of assets based on spatial proximity
pub struct AssetPreloader {
    /// Maps scene locations to lists of asset paths
    prediction_cache: HashMap<String, Vec<String>>,
    /// Queue of assets to be preloaded
    preload_queue: Arc<Mutex<VecDeque<String>>>,
    /// Set of assets currently being loaded
    loading_assets: Arc<Mutex<HashSet<String>>>,
    /// Radius within which to preload assets
    preload_radius: f32,
    /// Whether preloading is currently enabled
    enabled: AtomicBool,
}

impl Default for AssetPreloader {
    fn default() -> Self {
        Self {
            prediction_cache: HashMap::new(),
            preload_queue: Arc::new(Mutex::new(VecDeque::new())),
            loading_assets: Arc::new(Mutex::new(HashSet::new())),
            preload_radius: 100.0,
            enabled: AtomicBool::new(true),
        }
    }
}

impl AssetPreloader {
    /// Creates a new AssetPreloader with the specified preload radius
    pub fn new(preload_radius: f32) -> Self {
        Self {
            preload_radius,
            ..Default::default()
        }
    }

    /// Enable or disable preloading
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    /// Check if preloading is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Check if an asset is currently being loaded
    pub fn is_loading(&self, asset_path: &str) -> bool {
        self.loading_assets.lock().contains(asset_path)
    }

    /// Update the preload queue based on the current position
    pub fn update_preload_queue(
        &mut self,
        current_position: Vec3,
        scene_graph: &(impl SceneGraph + ?Sized),
    ) {
        if !self.enabled.load(Ordering::SeqCst) {
            return;
        }

        let nearby_assets = scene_graph.find_assets_near(current_position, self.preload_radius);
        let mut queue = self.preload_queue.lock();

        for asset_path in nearby_assets {
            if !self.is_loading(&asset_path) && !queue.iter().any(|p| p == &asset_path) {
                queue.push_back(asset_path);
            }
        }

        // Add predicted assets
        if let Some(predicted_assets) = self.prediction_cache.get(&current_position.to_string()) {
            for asset_path in predicted_assets {
                if !self.is_loading(asset_path) && !queue.iter().any(|p| p == asset_path) {
                    queue.push_back(asset_path.clone());
                }
            }
        }
    }

    /// Get the next asset to preload
    pub fn pop_next_asset(&self) -> Option<String> {
        self.preload_queue.lock().pop_front()
    }

    /// Mark an asset as loading
    pub fn mark_loading(&self, asset_path: &str) {
        self.loading_assets.lock().insert(asset_path.to_string());
    }

    /// Mark an asset as done loading
    pub fn mark_done(&self, asset_path: &str) {
        self.loading_assets.lock().remove(asset_path);
    }

    /// Add a prediction entry for a specific position
    pub fn add_prediction(&mut self, position: Vec3, likely_assets: Vec<String>) {
        self.prediction_cache
            .insert(position.to_string(), likely_assets);
    }

    /// Clear the preload queue
    pub fn clear_queue(&self) {
        self.preload_queue.lock().clear();
    }

    /// Set the preload radius
    pub fn set_preload_radius(&mut self, radius: f32) {
        self.preload_radius = radius.max(0.0);
    }

    /// Get the preload radius
    pub fn preload_radius(&self) -> f32 {
        self.preload_radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preloader_creation() {
        let preloader = AssetPreloader::new(50.0);
        assert_eq!(preloader.preload_radius(), 50.0);
        assert!(preloader.is_enabled());
    }

    #[test]
    fn test_preloader_enable_disable() {
        let preloader = AssetPreloader::new(100.0);
        assert!(preloader.is_enabled());

        preloader.set_enabled(false);
        assert!(!preloader.is_enabled());
    }
}
