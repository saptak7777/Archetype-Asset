//! Level of Detail (LOD) system for mesh optimization
//!
//! This module provides LOD management and mesh simplification for
//! optimizing rendering performance at different distances.

use crate::model::Mesh;
use std::sync::Arc;

/// Represents a model with multiple levels of detail
#[derive(Debug, Clone)]
pub struct LodModel {
    /// Different detail levels, from highest (index 0) to lowest quality
    pub lod_levels: Vec<Arc<Mesh>>,
    /// Screen space size thresholds for transitioning between LODs
    /// The length should be lod_levels.len() - 1
    pub transition_thresholds: Vec<f32>,
}

impl LodModel {
    /// Create a new LOD model with the given levels and thresholds
    pub fn new(lod_levels: Vec<Arc<Mesh>>, transition_thresholds: Vec<f32>) -> Self {
        assert!(
            !lod_levels.is_empty(),
            "LOD model must have at least one level of detail"
        );
        assert_eq!(
            lod_levels.len() - 1,
            transition_thresholds.len(),
            "Number of transition thresholds must be one less than number of LOD levels"
        );

        // Ensure thresholds are in descending order
        for i in 1..transition_thresholds.len() {
            assert!(
                transition_thresholds[i] < transition_thresholds[i - 1],
                "Transition thresholds must be in descending order"
            );
        }

        Self {
            lod_levels,
            transition_thresholds,
        }
    }

    /// Get the appropriate LOD level based on screen space size
    /// `screen_size` should be the height of the object in screen space (0.0 to 1.0)
    pub fn get_lod(&self, screen_size: f32) -> &Arc<Mesh> {
        for (i, &threshold) in self.transition_thresholds.iter().enumerate() {
            if screen_size >= threshold {
                return &self.lod_levels[i];
            }
        }
        self.lod_levels.last().unwrap() // Return lowest LOD
    }

    /// Get the LOD index for a given screen size
    pub fn get_lod_index(&self, screen_size: f32) -> usize {
        for (i, &threshold) in self.transition_thresholds.iter().enumerate() {
            if screen_size >= threshold {
                return i;
            }
        }
        self.lod_levels.len() - 1
    }

    /// Get the number of LOD levels
    pub fn level_count(&self) -> usize {
        self.lod_levels.len()
    }
}

/// Trait for mesh simplification algorithms
pub trait MeshSimplifier: Send + Sync {
    /// Simplify the mesh to approximately the given target vertex count
    /// Returns a new simplified mesh
    fn simplify(&self, mesh: &Mesh, target_vertex_count: usize) -> Mesh;

    /// Create a boxed clone of the simplifier
    fn box_clone(&self) -> Box<dyn MeshSimplifier + Send + Sync>;
}

/// Default implementation using basic vertex decimation
///
/// Note: For production use, enable the `lod-generation` feature
/// which uses the `meshopt` crate for proper mesh simplification.
#[derive(Clone, Copy)]
pub struct DefaultSimplifier;

impl Default for DefaultSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultSimplifier {
    /// Create a new default simplifier
    pub fn new() -> Self {
        Self
    }
}

impl MeshSimplifier for DefaultSimplifier {
    fn simplify(&self, mesh: &Mesh, target_vertex_count: usize) -> Mesh {
        let mesh_data = mesh.vertices();
        let current_vertex_count = mesh_data.vertices.len() / 8; // 8 floats per vertex

        // If already at or below target, return clone
        if current_vertex_count <= target_vertex_count {
            return mesh.clone();
        }

        // Simple decimation: skip every N vertices
        // This is a placeholder - real implementation would use proper edge collapse
        let skip_ratio = current_vertex_count / target_vertex_count.max(1);

        if skip_ratio <= 1 {
            return mesh.clone();
        }

        // For now, just return the original mesh
        // Real meshopt implementation would go here when feature is enabled
        mesh.clone()
    }

    fn box_clone(&self) -> Box<dyn MeshSimplifier + Send + Sync> {
        Box::new(*self)
    }
}

/// Meshopt-based simplifier for production use
#[cfg(feature = "lod")]
pub struct MeshoptSimplifier {
    /// Target error threshold (0.0 to 1.0)
    pub error_threshold: f32,
}

#[cfg(feature = "lod")]
impl MeshoptSimplifier {
    /// Create a new meshopt simplifier
    pub fn new(error_threshold: f32) -> Self {
        Self { error_threshold }
    }
}

#[cfg(feature = "lod")]
impl MeshSimplifier for MeshoptSimplifier {
    fn simplify(&self, mesh: &Mesh, target_vertex_count: usize) -> Mesh {
        use crate::model::PrimitiveType;
        use crate::renderer::Vertex;
        use meshopt::{SimplifyOptions, VertexDataAdapter};

        let mesh_data = mesh.vertices();
        let vertices = &mesh_data.vertices;
        let indices = &mesh_data.indices;

        // Convert to meshopt format
        let vertex_count = vertices.len() / 8;
        if target_vertex_count >= vertex_count {
            return mesh.clone();
        }

        // Create position array for meshopt (3 floats per vertex)
        let mut positions: Vec<f32> = Vec::with_capacity(vertex_count * 3);
        for i in 0..vertex_count {
            let base = i * 8;
            positions.push(vertices[base]); // x
            positions.push(vertices[base + 1]); // y
            positions.push(vertices[base + 2]); // z
        }

        // Create VertexDataAdapter for meshopt
        let position_bytes: &[u8] = bytemuck::cast_slice(&positions);
        let vertex_adapter = VertexDataAdapter::new(
            position_bytes,
            std::mem::size_of::<f32>() * 3, // stride: 3 floats per vertex
            0,                              // offset
        )
        .expect("Failed to create VertexDataAdapter");

        // Simplify using meshopt
        let target_index_count = (indices.len() * target_vertex_count) / vertex_count;

        let simplified_indices = meshopt::simplify(
            indices,
            &vertex_adapter,
            target_index_count.max(3),
            self.error_threshold,
            SimplifyOptions::empty(),
            None,
        );

        // Rebuild mesh with simplified indices
        // Collect used vertices
        let mut used_vertices: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &idx in &simplified_indices {
            used_vertices.insert(idx);
        }

        // Create vertex remap
        let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut new_vertices: Vec<Vertex> = Vec::new();

        for old_idx in used_vertices.iter() {
            let new_idx = new_vertices.len() as u32;
            remap.insert(*old_idx, new_idx);

            let base = (*old_idx as usize) * 8;
            new_vertices.push(Vertex {
                position: [vertices[base], vertices[base + 1], vertices[base + 2]],
                normal: [vertices[base + 3], vertices[base + 4], vertices[base + 5]],
                uv: [vertices[base + 6], vertices[base + 7]],
                color: [1.0, 1.0, 1.0], // Default color
            });
        }

        // Remap indices
        let new_indices: Vec<u32> = simplified_indices
            .iter()
            .map(|&idx| *remap.get(&idx).unwrap_or(&0))
            .collect();

        Mesh::new(
            mesh.name.clone(),
            PrimitiveType::Triangles,
            new_vertices,
            new_indices,
            mesh.material_index,
        )
    }

    fn box_clone(&self) -> Box<dyn MeshSimplifier + Send + Sync> {
        Box::new(Self {
            error_threshold: self.error_threshold,
        })
    }
}

/// Generate LOD levels for a mesh
///
/// # Arguments
/// * `mesh` - The source mesh (highest detail)
/// * `levels` - Number of LOD levels to generate (including source)
/// * `simplifier` - The simplification algorithm to use
///
/// # Returns
/// A vector of meshes from highest to lowest detail
pub fn generate_lod_levels(
    mesh: &Mesh,
    levels: usize,
    simplifier: &dyn MeshSimplifier,
) -> Vec<Arc<Mesh>> {
    if levels <= 1 {
        return vec![Arc::new(mesh.clone())];
    }

    let mesh_data = mesh.vertices();
    let vertex_count = mesh_data.vertices.len() / 8;
    let mut result = Vec::with_capacity(levels);

    // First level is the original mesh
    result.push(Arc::new(mesh.clone()));

    // Generate progressively simplified versions
    for i in 1..levels {
        // Each level has roughly half the vertices of the previous
        let target = vertex_count / (1 << i);
        let target = target.max(12); // Minimum 12 vertices (4 triangles)

        let simplified = simplifier.simplify(mesh, target);
        result.push(Arc::new(simplified));
    }

    result
}

/// Calculate default transition thresholds for N LOD levels
///
/// Uses a geometric progression from 0.5 down to smaller values
pub fn default_thresholds(levels: usize) -> Vec<f32> {
    if levels <= 1 {
        return vec![];
    }

    let mut thresholds = Vec::with_capacity(levels - 1);
    for i in 0..(levels - 1) {
        // Thresholds: 0.5, 0.25, 0.125, etc.
        thresholds.push(0.5 / (1 << i) as f32);
    }
    thresholds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PrimitiveType;
    use crate::renderer::Vertex;

    fn create_test_mesh() -> Mesh {
        // Create a simple quad mesh for testing
        let vertices = vec![
            Vertex {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0],
            },
        ];

        let indices = vec![0, 1, 2, 2, 3, 0];

        Mesh::new(
            Some("test_quad".to_string()),
            PrimitiveType::Triangles,
            vertices,
            indices,
            None,
        )
    }

    #[test]
    fn test_lod_selection() {
        let mesh = create_test_mesh();
        let lod_model = LodModel::new(
            vec![Arc::new(mesh.clone()), Arc::new(mesh.clone())],
            vec![0.5], // Single threshold at 50% screen size
        );

        // Test LOD selection
        assert!(Arc::ptr_eq(
            lod_model.get_lod(0.6),   // Above threshold
            &lod_model.lod_levels[0]  // Should return highest LOD
        ));

        assert!(Arc::ptr_eq(
            lod_model.get_lod(0.4),   // Below threshold
            &lod_model.lod_levels[1]  // Should return lowest LOD
        ));
    }

    #[test]
    fn test_lod_index() {
        let mesh = create_test_mesh();
        let lod_model = LodModel::new(
            vec![
                Arc::new(mesh.clone()),
                Arc::new(mesh.clone()),
                Arc::new(mesh.clone()),
            ],
            vec![0.5, 0.25],
        );

        assert_eq!(lod_model.get_lod_index(0.6), 0);
        assert_eq!(lod_model.get_lod_index(0.4), 1);
        assert_eq!(lod_model.get_lod_index(0.2), 2);
    }

    #[test]
    fn test_default_thresholds() {
        let thresholds = default_thresholds(4);
        assert_eq!(thresholds.len(), 3);
        assert!((thresholds[0] - 0.5).abs() < 0.001);
        assert!((thresholds[1] - 0.25).abs() < 0.001);
        assert!((thresholds[2] - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_generate_lod_levels() {
        let mesh = create_test_mesh();
        let simplifier = DefaultSimplifier::new();

        let levels = generate_lod_levels(&mesh, 3, &simplifier);
        assert_eq!(levels.len(), 3);
    }
}
