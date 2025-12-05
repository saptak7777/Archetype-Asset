//! Feature matrix tests - verify all feature combinations compile and work

use archetype_asset::{AssetCache, MockGpu};

#[test]
fn test_default_features() {
    // Default features: gpu-mock, runtime-mock
    let gpu = MockGpu::new();
    let cache = AssetCache::new(gpu, 1024);
    assert_eq!(cache.memory_usage(), 0);
}

#[test]
fn test_lod_types_available() {
    use archetype_asset::{default_thresholds, DefaultSimplifier};

    // Verify LOD types are available
    let _simplifier = DefaultSimplifier::new();
    let thresholds = default_thresholds(3);
    assert_eq!(thresholds.len(), 2);
}

#[test]
fn test_spatial_types_available() {
    use archetype_asset::{DistancePredictor, SpatialPredictor};
    use glam::Vec3;

    let predictor = DistancePredictor::new(100.0);
    assert_eq!(predictor.prediction_radius(), 100.0);

    // Test prediction
    let assets = predictor.predict_assets(Vec3::ZERO, None);
    assert!(assets.is_empty()); // Default returns empty
}

#[test]
fn test_memory_pool_types() {
    use archetype_asset::{AssetMemoryPool, MeshData};

    let pool = AssetMemoryPool::new();

    let mesh_data = MeshData {
        vertices: vec![1.0, 2.0, 3.0],
        indices: vec![0, 1, 2],
        material_index: None,
    };

    let _mesh = pool.alloc_mesh_data(mesh_data);
    assert_eq!(pool.mesh_data_count(), 1);
}

#[test]
fn test_gltf_loader_available() {
    use archetype_asset::loader::gltf::load_glb_bytes;

    // Empty data should fail gracefully
    let result = load_glb_bytes(&[]);
    assert!(result.is_err());
}
