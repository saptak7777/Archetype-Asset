//! Vertex types for rendering
//!
//! This module provides the vertex structure used throughout the asset system.

/// A vertex with position, normal, UV, tangent, and color data
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    /// 3D position
    pub position: [f32; 3],
    /// Normal vector
    pub normal: [f32; 3],
    /// Texture coordinates
    pub uv: [f32; 2],
    /// Tangent vector (xyz + w handedness)
    pub tangent: [f32; 4],
    /// Vertex color (RGBA)
    pub color: [f32; 4],
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0], // Default up (Z+)
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0], // Default tangent
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl Vertex {
    /// Create a new vertex
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
            tangent: [1.0, 0.0, 0.0, 1.0], // Default tangent
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Set the tangent vector
    pub fn with_tangent(mut self, tangent: [f32; 4]) -> Self {
        self.tangent = tangent;
        self
    }

    /// Create a vertex with color
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Size of a vertex in bytes
    pub const fn size() -> usize {
        std::mem::size_of::<Self>()
    }

    /// Number of floats per vertex
    pub const fn floats_per_vertex() -> usize {
        16 // 3 + 3 + 2 + 4 + 4
    }

    /// Convert to flat array of floats
    pub fn to_floats(&self) -> [f32; 16] {
        [
            self.position[0],
            self.position[1],
            self.position[2],
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.uv[0],
            self.uv[1],
            self.tangent[0],
            self.tangent[1],
            self.tangent[2],
            self.tangent[3],
            self.color[0],
            self.color[1],
            self.color[2],
            self.color[3],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_default() {
        let v = Vertex::default();
        assert_eq!(v.position, [0.0, 0.0, 0.0]);
        assert_eq!(v.normal, [0.0, 0.0, 1.0]);
        assert_eq!(v.tangent, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_vertex_size() {
        assert_eq!(Vertex::size(), 64);
        assert_eq!(Vertex::size(), std::mem::size_of::<Vertex>());
    }

    #[test]
    fn test_vertex_to_floats() {
        let v = Vertex::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.5, 0.5]);
        let floats = v.to_floats();
        assert_eq!(floats[0], 1.0);
        assert_eq!(floats[3], 0.0); // normal x
        assert_eq!(floats[8], 1.0); // tangent x
        assert_eq!(floats.len(), 16);
    }
}
