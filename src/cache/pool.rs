//! Memory pool implementation for efficient asset management
//!
//! This module provides arena-based memory pools for different asset types to
//! reduce allocation overhead and memory fragmentation.

use parking_lot::RwLock;
use shared_arena::{ArenaBox, SharedArena};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Represents the data stored for a texture in the memory pool
#[derive(Debug)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub format: crate::texture::TextureFormat,
    pub data: Vec<u8>,
}

/// Represents the vertex data stored in the memory pool
#[derive(Debug, PartialEq)]
pub struct VertexData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
}

/// Represents the mesh data stored in the memory pool
///
/// The vertices are stored as a flat Vec<f32> with the following layout:
/// [pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, u, v, ...]
/// Where:
/// - pos_*: 3D position coordinates
/// - norm_*: 3D normal vector
/// - u, v: 2D texture coordinates
#[derive(Debug, PartialEq)]
pub struct MeshData {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub material_index: Option<usize>,
}

/// A thread-safe memory pool for assets
#[derive(Debug)]
pub struct AssetMemoryPool {
    texture_arena: Arc<RwLock<SharedArena<TextureData>>>,
    vertex_arena: Arc<RwLock<SharedArena<VertexData>>>,
    mesh_arena: Arc<RwLock<SharedArena<MeshData>>>,
    texture_count: Arc<AtomicUsize>,
    vertex_data_count: Arc<AtomicUsize>,
    mesh_data_count: Arc<AtomicUsize>,
}

impl Default for AssetMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AssetMemoryPool {
    fn drop(&mut self) {
        // Ensure all arenas are properly dropped
        self.clear();
    }
}

impl AssetMemoryPool {
    /// Create a new memory pool with default settings
    pub fn new() -> Self {
        // Initialize each arena with a reasonable capacity
        let texture_arena = SharedArena::with_capacity(32);
        let vertex_arena = SharedArena::with_capacity(128);
        let mesh_arena = SharedArena::with_capacity(64);

        Self {
            texture_arena: Arc::new(RwLock::new(texture_arena)),
            vertex_arena: Arc::new(RwLock::new(vertex_arena)),
            mesh_arena: Arc::new(RwLock::new(mesh_arena)),
            texture_count: Arc::new(AtomicUsize::new(0)),
            vertex_data_count: Arc::new(AtomicUsize::new(0)),
            mesh_data_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Allocate texture data in the memory pool
    pub fn alloc_texture(&self, data: TextureData) -> ArenaBox<TextureData> {
        let arena = self.texture_arena.write();
        let result = arena.alloc(data);
        // Increment the counter
        self.texture_count.fetch_add(1, Ordering::SeqCst);
        result
    }

    /// Allocate vertex data in the memory pool
    pub fn alloc_vertex_data(&self, data: VertexData) -> ArenaBox<VertexData> {
        let arena = self.vertex_arena.write();
        let result = arena.alloc(data);
        // Increment the counter
        self.vertex_data_count.fetch_add(1, Ordering::SeqCst);
        result
    }

    /// Allocate mesh data in the memory pool
    pub fn alloc_mesh_data(&self, data: MeshData) -> ArenaBox<MeshData> {
        let arena = self.mesh_arena.write();
        let result = arena.alloc(data);
        // Increment the counter
        self.mesh_data_count.fetch_add(1, Ordering::SeqCst);
        result
    }

    /// Get the number of textures currently allocated
    pub fn texture_count(&self) -> usize {
        self.texture_count.load(Ordering::SeqCst)
    }

    /// Get the number of vertex data chunks currently allocated
    pub fn vertex_data_count(&self) -> usize {
        self.vertex_data_count.load(Ordering::SeqCst)
    }

    /// Get the number of mesh data chunks currently allocated
    pub fn mesh_data_count(&self) -> usize {
        self.mesh_data_count.load(Ordering::SeqCst)
    }

    /// Clear all allocated resources
    pub fn clear(&self) {
        // Clear each arena by replacing it with a new empty one
        *self.texture_arena.write() = SharedArena::new();
        *self.vertex_arena.write() = SharedArena::new();
        *self.mesh_arena.write() = SharedArena::new();

        // Reset all counters
        self.texture_count.store(0, Ordering::SeqCst);
        self.vertex_data_count.store(0, Ordering::SeqCst);
        self.mesh_data_count.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_allocation() {
        let pool = AssetMemoryPool::new();
        let texture_data = TextureData {
            width: 64,
            height: 64,
            format: crate::texture::TextureFormat::Rgba8,
            data: vec![0; 64 * 64 * 4],
        };

        let texture = pool.alloc_texture(texture_data);
        assert_eq!(texture.width, 64);
        assert_eq!(texture.height, 64);
        assert_eq!(pool.texture_count(), 1);
    }

    #[test]
    fn test_vertex_data_allocation() {
        let pool = AssetMemoryPool::new();
        let vertex_data = VertexData {
            positions: vec![[0.0, 0.0, 0.0]],
            normals: vec![[0.0, 1.0, 0.0]],
            tex_coords: vec![[0.0, 0.0]],
        };

        let vertices = pool.alloc_vertex_data(vertex_data);
        assert_eq!(vertices.positions.len(), 1);
        assert_eq!(pool.vertex_data_count(), 1);
    }

    #[test]
    fn test_mesh_data_allocation() {
        let pool = AssetMemoryPool::new();

        // Create a simple triangle with position data only (3 vertices, 3 components each)
        let vertices = vec![
            // Vertex 1
            0.0, 0.0, 0.0, // Vertex 2
            1.0, 0.0, 0.0, // Vertex 3
            0.0, 1.0, 0.0,
        ];

        let mesh_data = MeshData {
            vertices,
            indices: vec![0, 1, 2], // triangle indices
            material_index: Some(0),
        };

        let mesh = pool.alloc_mesh_data(mesh_data);
        assert_eq!(mesh.vertices.len(), 9); // 3 vertices * 3 components
        assert_eq!(mesh.indices.len(), 3); // 3 indices for a triangle
        assert_eq!(pool.mesh_data_count(), 1);
    }
}
