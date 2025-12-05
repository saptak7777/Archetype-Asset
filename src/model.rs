use crate::cache::pool::{AssetMemoryPool, MeshData};
use crate::renderer::Vertex;
use futures::future::join_all;
use gltf::Gltf;
use parking_lot::RwLock;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// Type of primitive to render
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

/// Error type for model loading operations
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("Unsupported model format: {0}")]
    UnsupportedFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GLTF error: {0}")]
    Gltf(#[from] gltf::Error),

    #[error("Unsupported primitive mode")]
    UnsupportedPrimitiveMode,
}

/// How to handle transparency
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// Supported texture formats with compression and quality characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// Uncompressed RGBA with sRGB color space
    R8G8B8A8Srgb,

    /// WebP compressed texture (25-35% smaller than JPEG with better quality)
    WebP,

    /// KTX2 with ETC1S compression (60-80% smaller, fast GPU upload)
    Ktx2Etc1s,

    /// KTX2 with UASTC compression (40-60% smaller, excellent quality)
    Ktx2Uastc,

    /// Legacy RGB format (no alpha)
    R8G8B8Srgb,

    /// Legacy single-channel format
    R8Srgb,
}

impl TextureFormat {
    /// Get the file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::R8G8B8A8Srgb | Self::R8G8B8Srgb | Self::R8Srgb => "png",
            Self::WebP => "webp",
            Self::Ktx2Etc1s | Self::Ktx2Uastc => "ktx2",
        }
    }

    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::R8G8B8A8Srgb | Self::R8G8B8Srgb | Self::R8Srgb => "image/png",
            Self::WebP => "image/webp",
            Self::Ktx2Etc1s | Self::Ktx2Uastc => "image/ktx2",
        }
    }

    /// Check if this format is compressed
    pub fn is_compressed(&self) -> bool {
        !matches!(self, Self::R8G8B8A8Srgb | Self::R8G8B8Srgb | Self::R8Srgb)
    }

    /// Get the estimated size reduction compared to uncompressed RGBA
    pub fn size_reduction(&self) -> f32 {
        match self {
            Self::R8G8B8A8Srgb => 1.0,
            Self::R8G8B8Srgb => 0.75, // 25% smaller (no alpha)
            Self::R8Srgb => 0.25,     // 75% smaller (single channel)
            Self::WebP => 0.7,        // 30% smaller
            Self::Ktx2Etc1s => 0.3,   // 70% smaller
            Self::Ktx2Uastc => 0.5,   // 50% smaller
        }
    }

    /// Get the relative GPU upload speed (higher is better)
    pub fn upload_speed(&self) -> f32 {
        match self {
            Self::R8G8B8A8Srgb | Self::R8G8B8Srgb | Self::R8Srgb => 1.0,
            Self::WebP => 1.2,
            Self::Ktx2Etc1s => 3.0,
            Self::Ktx2Uastc => 2.5,
        }
    }

    /// Get the recommended format based on quality settings
    pub fn recommended(quality: TextureQuality) -> Self {
        match quality {
            TextureQuality::Fastest => Self::Ktx2Etc1s,
            TextureQuality::Balanced => Self::Ktx2Uastc,
            TextureQuality::Best => Self::WebP,
        }
    }
}

/// Material properties for PBR rendering
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    /// Optional name of the material
    pub name: Option<String>,

    // Base color
    /// Base color factor (RGBA)
    pub base_color_factor: [f32; 4],
    /// Base color texture index
    pub base_color_texture: Option<usize>,
    /// Texture coordinate set for base color texture
    pub base_color_texture_tex_coord: u32,

    // Metallic-roughness
    /// Metallic factor
    pub metallic_factor: f32,
    /// Roughness factor
    pub roughness_factor: f32,
    /// Metallic-roughness texture (B: metallic, G: roughness)
    pub metallic_roughness_texture: Option<usize>,
    /// Texture coordinate set for metallic-roughness texture
    pub metallic_roughness_tex_coord: u32,

    // Normal map
    /// Normal map texture index
    pub normal_texture: Option<usize>,
    /// Normal scale factor
    pub normal_scale: f32,
    /// Texture coordinate set for normal map
    pub normal_tex_coord: u32,

    // Occlusion
    /// Occlusion texture index
    pub occlusion_texture: Option<usize>,
    /// Occlusion strength
    pub occlusion_strength: f32,
    /// Texture coordinate set for occlusion texture
    pub occlusion_tex_coord: u32,

    // Emissive
    /// Emissive texture index
    pub emissive_texture: Option<usize>,
    /// Emissive factor (RGB)
    pub emissive_factor: [f32; 3],
    /// Texture coordinate set for emissive texture
    pub emissive_tex_coord: u32,

    // Alpha
    /// Alpha mode
    pub alpha_mode: AlphaMode,
    /// Alpha cutoff for masked blending
    pub alpha_cutoff: f32,
    /// Whether the material is double-sided
    pub double_sided: bool,
    /// Whether the material is unlit (KHR_materials_unlit extension)
    pub unlit: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: None,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            base_color_texture_tex_coord: 0,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            metallic_roughness_tex_coord: 0,
            normal_texture: None,
            normal_scale: 1.0,
            normal_tex_coord: 0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            occlusion_tex_coord: 0,
            emissive_texture: None,
            emissive_factor: [0.0, 0.0, 0.0],
            emissive_tex_coord: 0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
        }
    }
}

/// A 3D mesh with vertex data and material index
#[derive(Clone)]
pub struct Mesh {
    /// Optional name of the mesh
    pub name: Option<String>,
    /// Type of primitive (triangles, lines, etc.)
    pub primitive_type: PrimitiveType,
    /// Reference to the mesh data in the memory pool
    mesh_data: Arc<RwLock<MeshData>>,
    /// Index of the material used by this mesh
    pub material_index: Option<usize>,
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("name", &self.name)
            .field("primitive_type", &self.primitive_type)
            .field("material_index", &self.material_index)
            .field("vertex_count", &self.vertex_count())
            .field("index_count", &self.index_count())
            .finish()
    }
}

impl Mesh {
    /// Create a new mesh with the given data
    pub fn new(
        name: Option<String>,
        primitive_type: PrimitiveType,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        material_index: Option<usize>,
    ) -> Self {
        // Convert Vec<Vertex> to Vec<f32> for storage in MeshData
        // Pre-allocate exact size to avoid reallocations
        let vertex_count = vertices.len();
        let mut vertex_data = Vec::with_capacity(vertex_count * 16); // 16 floats per vertex

        for v in vertices {
            vertex_data.extend_from_slice(&v.position);
            vertex_data.extend_from_slice(&v.normal);
            vertex_data.extend_from_slice(&v.uv);
            vertex_data.extend_from_slice(&v.tangent);
            vertex_data.extend_from_slice(&v.color);
        }

        let mesh_data = MeshData {
            vertices: vertex_data,
            indices,
            material_index,
        };

        Self {
            name,
            primitive_type,
            mesh_data: Arc::new(RwLock::new(mesh_data)),
            material_index,
        }
    }

    /// Get a read-only reference to the vertex data
    pub fn vertices(&self) -> parking_lot::RwLockReadGuard<'_, MeshData> {
        self.mesh_data.read()
    }

    /// Get a mutable reference to the vertex data
    pub fn vertices_mut(&self) -> parking_lot::RwLockWriteGuard<'_, MeshData> {
        self.mesh_data.write()
    }

    /// Get the number of vertices in the mesh
    pub fn vertex_count(&self) -> usize {
        self.vertices().vertices.len()
    }

    /// Get the number of vertices in the mesh (alias for vertex_count)
    pub fn get_vertex_count(&self) -> usize {
        self.vertex_count()
    }

    /// Get the number of indices in the mesh
    pub fn index_count(&self) -> usize {
        self.vertices().indices.len()
    }

    /// Get the vertex buffer as a slice of bytes
    pub fn vertex_buffer(&self) -> Vec<u8> {
        let data = self.vertices();
        // Convert the slice to bytes and create a new Vec<u8> to own the data
        bytemuck::cast_slice(&data.vertices).to_vec()
    }

    /// Get the index buffer as a slice of bytes
    pub fn index_buffer(&self) -> Vec<u8> {
        let data = self.vertices();
        // Convert the slice to bytes and create a new Vec<u8> to own the data
        bytemuck::cast_slice(&data.indices).to_vec()
    }

    /// Get a copy of the vertex data as a Vec<f32>
    pub fn vertex_data(&self) -> Vec<f32> {
        let data = self.vertices();
        data.vertices.clone()
    }

    /// Get a copy of the index data as a Vec<u32>
    pub fn index_data(&self) -> Vec<u32> {
        let data = self.vertices();
        data.indices.clone()
    }
}

impl PartialEq for Mesh {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name &&
        self.primitive_type == other.primitive_type &&
        self.material_index == other.material_index &&
        // Compare the actual mesh data, not the Arc
        *self.mesh_data.read() == *other.mesh_data.read()
    }
}

/// Spatial transform (translation, rotation, scale)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4], // quaternion (x, y, z, w)
    pub scale: [f32; 3],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0], // identity quaternion
            scale: [1.0; 3],
        }
    }
}

/// A node in the scene hierarchy
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Optional name of the node
    pub name: Option<String>,
    /// Local transform of the node
    pub transform: Transform,
    /// Indices of meshes attached to this node
    pub mesh_indices: Vec<usize>,
    /// Indices of child nodes
    pub children: Vec<usize>,
}

/// Handle to a loaded model
#[derive(Debug, Clone)]
pub struct ModelHandle {
    #[allow(dead_code)]
    id: Uuid,
    model: Option<Arc<RwLock<LoadedModel>>>,
}

impl Default for ModelHandle {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            model: None,
        }
    }
}

impl ModelHandle {
    /// Create a new model handle
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the loaded model data
    pub fn set_model(&mut self, model: LoadedModel) {
        self.model = Some(Arc::new(RwLock::new(model)));
    }

    /// Get the model if it's fully loaded
    pub fn get_model(&self) -> Option<LoadedModel> {
        self.model.as_ref().map(|m| m.read().clone())
    }
}

/// Represents a loaded 3D model
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedModel {
    /// List of meshes in the model
    pub meshes: Vec<Mesh>,
    /// List of materials used by the meshes
    pub materials: Vec<Material>,
    /// List of nodes in the scene graph
    pub nodes: Vec<Node>,
    /// List of textures used by the materials
    pub textures: Vec<Texture>,
    /// The root nodes of the scene
    pub scene_root: Option<usize>,
}

use crate::lod::{DefaultSimplifier, LodModel, MeshSimplifier};

/// Texture compression quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureQuality {
    /// Fastest loading, lowest quality
    Fastest,
    /// Balanced quality and performance
    Balanced,
    /// Best quality, slower loading
    Best,
}

impl Default for TextureQuality {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Texture loading configuration
#[derive(Debug, Clone)]
pub struct TextureLoadConfig {
    /// Target quality level
    pub quality: TextureQuality,
    /// Maximum texture dimension (width/height)
    pub max_dimension: u32,
    /// Whether to generate mipmaps
    pub generate_mipmaps: bool,
    /// Whether to prefer sRGB color space
    pub srgb: bool,
}

impl Default for TextureLoadConfig {
    fn default() -> Self {
        Self {
            quality: TextureQuality::default(),
            max_dimension: 4096,
            generate_mipmaps: true,
            srgb: true,
        }
    }
}

/// A texture used in materials
#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    /// Optional name of the texture
    pub name: Option<String>,
    /// Width of the texture in pixels
    pub width: u32,
    /// Height of the texture in pixels
    pub height: u32,
    /// Format of the texture data
    pub format: TextureFormat,
    /// Raw texture data
    pub data: Vec<u8>,
    /// Whether the texture uses sRGB color space
    pub srgb: bool,
    /// Whether mipmaps should be generated for this texture
    pub generate_mipmaps: bool,
    /// Quality setting used when loading/processing this texture
    pub quality: TextureQuality,
}

impl Texture {
    /// Create a new texture with the given data and format
    pub fn new(
        name: Option<String>,
        width: u32,
        height: u32,
        format: TextureFormat,
        data: Vec<u8>,
        config: TextureLoadConfig,
    ) -> Self {
        Self {
            name,
            width,
            height,
            format,
            data,
            srgb: config.srgb,
            generate_mipmaps: config.generate_mipmaps,
            quality: config.quality,
        }
    }

    /// Create a new texture with recommended settings based on quality
    pub fn with_quality(
        name: Option<String>,
        width: u32,
        height: u32,
        data: Vec<u8>,
        quality: TextureQuality,
    ) -> Self {
        let format = TextureFormat::recommended(quality);
        let config = TextureLoadConfig {
            quality,
            max_dimension: width.max(height),
            generate_mipmaps: true,
            srgb: true,
        };

        Self::new(name, width, height, format, data, config)
    }

    /// Get the estimated size of this texture in bytes
    pub fn estimated_size(&self) -> usize {
        let base_size = self.width as f32 * self.height as f32 * 4.0; // RGBA8 as baseline
        (base_size * self.format.size_reduction()) as usize
    }

    /// Get the expected GPU upload speed for this texture
    pub fn upload_speed(&self) -> f32 {
        self.format.upload_speed()
    }
}

/// Texture compression quality settings
///
/// Loads 3D models from various file formats
pub struct ModelLoader {
    memory_pool: Arc<RwLock<AssetMemoryPool>>,
    simplifier: Box<dyn MeshSimplifier + Send + Sync>,
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self {
            memory_pool: Arc::new(RwLock::new(AssetMemoryPool::new())),
            simplifier: Box::new(DefaultSimplifier),
        }
    }
}

// Implement Clone manually since we have a trait object
impl Clone for ModelLoader {
    fn clone(&self) -> Self {
        Self {
            memory_pool: self.memory_pool.clone(),
            simplifier: self.simplifier.box_clone(),
        }
    }
}

// Implement Debug manually for ModelLoader
impl std::fmt::Debug for ModelLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelLoader")
            .field("memory_pool", &self.memory_pool)
            .field("simplifier", &"<mesh simplifier>")
            .finish()
    }
}

impl ModelLoader {
    /// Create a new model loader with a reference to the memory pool
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a GLB model with optimized parallel processing and deferred texture loading
    pub async fn load_gltf_optimized(&self, data: &[u8]) -> Result<LoadedModel, ModelError> {
        // Parse the GLTF document
        let gltf = Gltf::from_slice(data)?;

        // Process materials, meshes, and nodes in parallel
        let (materials_future, meshes_future, nodes_future) = futures::join!(
            self.load_materials_async(&gltf),
            self.load_meshes_async(&gltf),
            self.load_nodes_async(&gltf)
        );

        // Wait for all parallel operations to complete
        let materials = materials_future?;
        let meshes = meshes_future?;
        let nodes = nodes_future?;

        // Create the model with empty textures (deferred loading)
        Ok(LoadedModel {
            meshes,
            materials,
            nodes,
            textures: Vec::new(), // Textures will be loaded on-demand
            scene_root: gltf
                .default_scene()
                .and_then(|s| s.nodes().next())
                .map(|n| n.index()),
        })
    }

    /// Asynchronously load materials from a GLTF document
    async fn load_materials_async(&self, gltf: &Gltf) -> Result<Vec<Material>, ModelError> {
        let materials = gltf
            .materials()
            .map(|mat| self.process_material(&mat))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(materials)
    }

    /// Asynchronously load meshes from a GLTF document
    async fn load_meshes_async(&self, gltf: &Gltf) -> Result<Vec<Mesh>, ModelError> {
        // Collect meshes into a Vec to avoid borrowing issues
        let meshes: Vec<_> = gltf.meshes().collect();

        // Process meshes in parallel chunks
        let mesh_futures = meshes.into_iter().map(|mesh| self.process_mesh_async(mesh));

        // Wait for all mesh processing to complete
        let meshes = join_all(mesh_futures)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        Ok(meshes)
    }

    /// Process a single mesh asynchronously
    async fn process_mesh_async<'a>(&self, mesh: gltf::Mesh<'a>) -> Result<Mesh, ModelError> {
        // This is a simplified version - you'll need to implement the actual mesh processing
        // based on your existing load_glb implementation
        let name = mesh.name().map(|s| s.to_string());

        // Collect primitives into a Vec to avoid borrowing issues
        let primitives: Vec<_> = mesh.primitives().collect();

        // Process primitives in parallel
        let _primitive_futures: Vec<_> = primitives
            .into_iter()
            .map(|prim| self.process_primitive_async(prim))
            .collect();

        // For now, just take the first primitive
        if let Some(primitive) = mesh.primitives().next() {
            // Store material index before moving the primitive
            let material_index = primitive.material().index();
            let (vertices, indices) = self.process_primitive_async(primitive).await?;

            Ok(Mesh::new(
                name,
                PrimitiveType::Triangles, // Default to triangles
                vertices,
                indices,
                material_index,
            ))
        } else {
            Err(ModelError::LoadError("Mesh has no primitives".to_string()))
        }
    }

    /// Process a single primitive asynchronously
    async fn process_primitive_async<'a>(
        &self,
        primitive: gltf::Primitive<'a>,
    ) -> Result<(Vec<Vertex>, Vec<u32>), ModelError> {
        // Removed unused import
        use crate::renderer::Vertex;
        use glam::Vec3;

        let mut vertices = Vec::new();
        // Get the reader for the primitive's vertex data
        let reader = primitive.reader(|_buffer| {
            // This closure is called to get the buffer data
            // For now, we'll just return an empty slice since we're using the buffer views directly
            Some(&[][..])
        });

        // Read positions (required)
        let positions: Vec<[f32; 3]> = if let Some(iter) = reader.read_positions() {
            iter.collect()
        } else {
            return Err(ModelError::LoadError("Mesh has no positions".to_string()));
        };

        // Read normals (optional, generate if not present)
        let normals: Vec<[f32; 3]> = if let Some(iter) = reader.read_normals() {
            iter.collect()
        } else {
            // Generate flat normals if none provided
            // Initialize with zero for accumulation
            let mut normals = vec![[0.0; 3]; positions.len()];

            // If we have indices, use them to generate smooth normals
            if let Some(iter) = reader.read_indices() {
                let indices: Vec<u32> = iter.into_u32().collect();

                // Generate smooth normals by averaging face normals
                for chunk in indices.chunks_exact(3) {
                    let i0 = chunk[0] as usize;
                    let i1 = chunk[1] as usize;
                    let i2 = chunk[2] as usize;

                    let v0 = Vec3::from_array(positions[i0]);
                    let v1 = Vec3::from_array(positions[i1]);
                    let v2 = Vec3::from_array(positions[i2]);

                    let edge1 = v1 - v0;
                    let edge2 = v2 - v0;
                    let normal = edge1.cross(edge2);

                    // Only add if not degenerate
                    if normal.length_squared() > 1e-6 {
                        let normal = normal.normalize();
                        // Add this normal to all three vertices
                        for &i in chunk {
                            let idx = i as usize;
                            normals[idx][0] += normal.x;
                            normals[idx][1] += normal.y;
                            normals[idx][2] += normal.z;
                        }
                    }
                }

                // Normalize all normals
                for normal in &mut normals {
                    let n = Vec3::from_array(*normal);
                    if n.length_squared() > 1e-6 {
                        *normal = n.normalize().to_array();
                    } else {
                        // Fallback for isolated vertices or degenerate geometry
                        *normal = [0.0, 0.0, 1.0]; // Default Up
                    }
                }
            } else {
                // No indices, generate flat normals for triangle list
                for chunk in positions.chunks_exact(3) {
                    let v0 = Vec3::from_array(chunk[0]);
                    let v1 = Vec3::from_array(chunk[1]);
                    let v2 = Vec3::from_array(chunk[2]);

                    let edge1 = v1 - v0;
                    let edge2 = v2 - v0;
                    let normal = edge1.cross(edge2);

                    let normal_array = if normal.length_squared() > 1e-6 {
                        normal.normalize().to_array()
                    } else {
                        [0.0, 0.0, 1.0]
                    };

                    for _ in 0..3 {
                        normals.push(normal_array);
                    }
                }
            }

            normals
        };

        // Read tangents (optional, default if not present)
        let tangents: Vec<[f32; 4]> = if let Some(iter) = reader.read_tangents() {
            iter.collect()
        } else {
            // Default tangent (x-axis)
            vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]
        };

        // Read colors (optional, default to white)
        let colors: Vec<[f32; 4]> = if let Some(iter) = reader.read_colors(0) {
            iter.into_rgba_f32().collect()
        } else {
            vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
        };

        // Read texture coordinates (required for PBR, default to zeros if not present)
        let tex_coords = if let Some(iter) = reader.read_tex_coords(0) {
            iter.into_f32().collect::<Vec<_>>()
        } else {
            // If no texture coordinates, generate default ones
            vec![[0.0, 0.0]; positions.len()]
        };

        // Create vertices with all attributes
        for i in 0..positions.len() {
            vertices.push(Vertex {
                position: positions[i],
                normal: normals[i],
                uv: tex_coords.get(i).copied().unwrap_or([0.0, 0.0]),
                tangent: tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
                color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
            });
        }

        // Read indices if available, otherwise generate them
        let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
            iter.into_u32().collect()
        } else {
            // If no indices, generate a simple triangle list
            (0..vertices.len() as u32).collect()
        };

        log::debug!(
            "Loaded primitive with {} vertices and {} indices",
            vertices.len(),
            indices.len()
        );

        Ok((vertices, indices))
    }

    /// Process a material (synchronous for now, but could be made async if needed)
    fn process_material(&self, _material: &gltf::Material) -> Result<Material, ModelError> {
        // Implement material processing based on your existing implementation
        // This is just a placeholder
        Ok(Material::default())
    }

    /// Asynchronously load nodes from a GLTF document
    async fn load_nodes_async(&self, gltf: &Gltf) -> Result<Vec<Node>, ModelError> {
        // Collect nodes into a Vec to avoid borrowing issues
        let nodes: Vec<_> = gltf.nodes().collect();

        // Process nodes in parallel
        let node_futures = nodes.into_iter().map(|node| self.process_node_async(node));

        // Wait for all node processing to complete
        let nodes = join_all(node_futures)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        Ok(nodes)
    }

    /// Process a single node asynchronously
    async fn process_node_async<'a>(&self, node: gltf::Node<'a>) -> Result<Node, ModelError> {
        // Get the transform from the GLTF node
        let (translation, rotation, scale) = node.transform().decomposed();

        // Create the transform
        let transform = Transform {
            translation: [translation[0], translation[1], translation[2]],
            rotation: [rotation[0], rotation[1], rotation[2], rotation[3]],
            scale: [scale[0], scale[1], scale[2]],
        };

        // Get mesh indices for this node
        let mesh_indices = if let Some(mesh) = node.mesh() {
            vec![mesh.index()]
        } else {
            Vec::new()
        };

        // Get child node indices (they'll be processed separately)
        let children = node.children().map(|c| c.index()).collect();

        Ok(Node {
            name: node.name().map(|s| s.to_string()),
            transform,
            mesh_indices,
            children,
        })
    }

    /// Create a new model loader with a specific memory pool
    pub fn with_memory_pool(memory_pool: Arc<RwLock<AssetMemoryPool>>) -> Self {
        Self {
            memory_pool,
            simplifier: Box::new(DefaultSimplifier),
        }
    }

    /// Create a new ModelLoader with a custom mesh simplifier
    pub fn with_simplifier(
        memory_pool: Arc<RwLock<AssetMemoryPool>>,
        simplifier: impl MeshSimplifier + 'static,
    ) -> Self {
        Self {
            memory_pool,
            simplifier: Box::new(simplifier),
        }
    }

    /// Generate LODs for a mesh
    ///
    /// # Arguments
    /// * `base_mesh` - The highest detail level mesh
    /// * `levels` - Number of LOD levels to generate (including the base level)
    /// * `reduction_factor` - How much to reduce the triangle count by at each LOD level (0.0 to 1.0)
    ///
    /// # Returns
    /// A LodModel containing all LOD levels and transition thresholds
    pub fn generate_lods(&self, base_mesh: &Mesh, levels: u32, reduction_factor: f32) -> LodModel {
        let mut lod_levels = vec![Arc::new(base_mesh.clone())];

        for level in 1..levels {
            let target_vertices = (base_mesh.get_vertex_count() as f32
                * reduction_factor.powi(level as i32)) as usize;
            let simplified_mesh = self.simplifier.simplify(base_mesh, target_vertices);
            lod_levels.push(Arc::new(simplified_mesh));
        }

        // Create transition thresholds (screen space size from 1.0 to 0.0)
        let transition_thresholds: Vec<f32> = (1..levels)
            .map(|i| 1.0 - (i as f32 / levels as f32))
            .collect();

        LodModel::new(lod_levels, transition_thresholds)
    }

    /// Load a texture from raw data with the specified configuration
    ///
    /// This method will automatically convert the texture to the desired format
    /// based on the provided configuration.
    pub async fn load_texture(
        &self,
        name: Option<String>,
        width: u32,
        height: u32,
        format: TextureFormat,
        data: Vec<u8>,
        config: TextureLoadConfig,
    ) -> Result<Texture, ModelError> {
        // In a real implementation, this would handle format conversion,
        // mipmap generation, and other processing asynchronously
        Ok(Texture::new(name, width, height, format, data, config))
    }

    /// Load a texture with automatic format selection based on quality settings
    pub async fn load_texture_with_quality(
        &self,
        name: Option<String>,
        width: u32,
        height: u32,
        data: Vec<u8>,
        quality: TextureQuality,
    ) -> Result<Texture, ModelError> {
        let format = TextureFormat::recommended(quality);
        let config = TextureLoadConfig {
            quality,
            max_dimension: width.max(height),
            generate_mipmaps: true,
            srgb: true,
        };

        self.load_texture(name, width, height, format, data, config)
            .await
    }

    /// Load a GLB model from binary data
    pub fn load_glb(&self, data: &[u8]) -> Result<LoadedModel, ModelError> {
        // Parse the GLB file
        let gltf = Gltf::from_slice(data)
            .map_err(|e| ModelError::LoadError(format!("Failed to parse GLB: {e}")))?;

        // Ensure we have binary data
        let blob = gltf
            .blob
            .as_ref()
            .ok_or_else(|| ModelError::LoadError("GLB file is missing binary data".to_string()))?;

        log::debug!(
            "Successfully parsed GLB with {} meshes and {} materials",
            gltf.meshes().len(),
            gltf.materials().len()
        );

        // Load materials with PBR support
        let materials: Vec<Material> = gltf
            .materials()
            .enumerate()
            .map(|(i, material)| {
                log::debug!(
                    "Loading material {i}: {:?}",
                    material.name().unwrap_or("unnamed")
                );

                let pbr = material.pbr_metallic_roughness();
                let alpha_mode = match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                };

                let mut mat = Material {
                    name: material.name().map(|s| s.to_string()),
                    double_sided: material.double_sided(),
                    alpha_mode,
                    alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
                    base_color_factor: pbr.base_color_factor(),
                    ..Default::default()
                };
                if let Some(info) = pbr.base_color_texture() {
                    mat.base_color_texture = Some(info.texture().index());
                    mat.base_color_texture_tex_coord = info.tex_coord();
                    log::debug!(
                        "  - Base color texture: {} (tex coord: {})",
                        mat.base_color_texture.unwrap(),
                        mat.base_color_texture_tex_coord
                    );
                }

                // Metallic-roughness
                mat.metallic_factor = pbr.metallic_factor();
                mat.roughness_factor = pbr.roughness_factor();
                if let Some(info) = pbr.metallic_roughness_texture() {
                    mat.metallic_roughness_texture = Some(info.texture().index());
                    mat.metallic_roughness_tex_coord = info.tex_coord();
                }

                // Normal map
                if let Some(normal) = material.normal_texture() {
                    mat.normal_texture = Some(normal.texture().index());
                    mat.normal_scale = normal.scale();
                    mat.normal_tex_coord = normal.tex_coord();
                }

                // Occlusion
                if let Some(occlusion) = material.occlusion_texture() {
                    mat.occlusion_texture = Some(occlusion.texture().index());
                    mat.occlusion_strength = occlusion.strength();
                    mat.occlusion_tex_coord = occlusion.tex_coord();
                }

                // Emission
                mat.emissive_factor = material.emissive_factor();
                if let Some(emissive) = material.emissive_texture() {
                    mat.emissive_texture = Some(emissive.texture().index());
                    mat.emissive_tex_coord = emissive.tex_coord();
                }

                // Check for KHR_materials_unlit extension (not directly supported in this version)
                // We'll set unlit to false by default as the GLTF crate version doesn't directly support the extension
                mat.unlit = false;

                log::debug!("  - Material loaded: {:?} (unlit: {})", mat.name, mat.unlit);
                mat
            })
            .collect();

        // Load meshes
        let mut meshes = Vec::new();

        for (mesh_idx, mesh) in gltf.meshes().enumerate() {
            log::debug!(
                "Loading mesh {mesh_idx}: {}",
                mesh.name().unwrap_or("unnamed")
            );

            for (prim_idx, primitive) in mesh.primitives().enumerate() {
                log::debug!("  - Processing primitive {prim_idx}");

                // Handle different primitive types
                let primitive_type = match primitive.mode() {
                    gltf::mesh::Mode::Points => PrimitiveType::Points,
                    gltf::mesh::Mode::Lines => PrimitiveType::Lines,
                    gltf::mesh::Mode::LineLoop => {
                        log::warn!(
                            "Line loop primitive mode is not supported, converting to line strip"
                        );
                        PrimitiveType::LineStrip
                    }
                    gltf::mesh::Mode::LineStrip => PrimitiveType::LineStrip,
                    gltf::mesh::Mode::Triangles => PrimitiveType::Triangles,
                    gltf::mesh::Mode::TriangleStrip => PrimitiveType::TriangleStrip,
                    gltf::mesh::Mode::TriangleFan => PrimitiveType::TriangleFan,
                };

                // Create a reader for the primitive data
                let reader = primitive.reader(|buffer| {
                    let view = gltf.views().nth(buffer.index()).unwrap();
                    let start = view.offset();
                    let end = start + view.length();

                    if end > blob.len() {
                        log::error!("Buffer view out of bounds: {end} > {}", blob.len());
                        return None;
                    }

                    Some(&blob[start..end])
                });

                // Extract positions (required)
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or_else(|| {
                        let err =
                            format!("Mesh {mesh_idx} primitive {prim_idx} is missing positions");
                        log::error!("{err}");
                        ModelError::LoadError(err)
                    })?
                    .collect();

                log::debug!("    - Found {} vertices", positions.len());

                // Extract normals (generate if missing)
                let normals = if let Some(normals_iter) = reader.read_normals() {
                    normals_iter.collect::<Vec<[f32; 3]>>()
                } else {
                    log::debug!("    - Generating flat normals");
                    let mut normals = vec![[0.0, 0.0, 0.0]; positions.len()];

                    if let Some(indices) = reader.read_indices().map(|i| i.into_u32()) {
                        let indices = indices.collect::<Vec<_>>();

                        // Generate smooth normals by averaging face normals
                        for chunk in indices.chunks(3) {
                            if chunk.len() == 3 {
                                let i0 = chunk[0] as usize;
                                let i1 = chunk[1] as usize;
                                let i2 = chunk[2] as usize;

                                if i0 >= positions.len()
                                    || i1 >= positions.len()
                                    || i2 >= positions.len()
                                {
                                    log::warn!("Invalid vertex index in mesh {mesh_idx} primitive {prim_idx}");
                                    continue;
                                }

                                let v0 = glam::Vec3::from(positions[i0]);
                                let v1 = glam::Vec3::from(positions[i1]);
                                let v2 = glam::Vec3::from(positions[i2]);

                                let edge1 = v1 - v0;
                                let edge2 = v2 - v0;
                                let normal = edge1.cross(edge2);

                                if normal.length_squared() > 1e-6 {
                                    let normal = normal.normalize();
                                    // Add to each vertex's normal (accumulate)
                                    normals[i0] = (glam::Vec3::from(normals[i0]) + normal).into();
                                    normals[i1] = (glam::Vec3::from(normals[i1]) + normal).into();
                                    normals[i2] = (glam::Vec3::from(normals[i2]) + normal).into();
                                }
                            }
                        }

                        // Normalize accumulated normals
                        for normal in &mut normals {
                            let n = glam::Vec3::from(*normal);
                            if n.length_squared() > 1e-6 {
                                *normal = n.normalize().into();
                            } else {
                                *normal = [0.0, 0.0, 1.0]; // Default Up
                            }
                        }
                    }

                    normals
                };

                // Extract texture coordinates (default to zero if missing)
                // Extract texture coordinates (default to zero if missing)
                let tex_coords = if let Some(tex_coord_iter) = reader.read_tex_coords(0) {
                    tex_coord_iter.into_f32().collect::<Vec<[f32; 2]>>()
                } else {
                    log::debug!("    - No texture coordinates found, using defaults");
                    vec![[0.0, 0.0]; positions.len()]
                };

                // Extract tangents (default if missing)
                let tangents = if let Some(iter) = reader.read_tangents() {
                    iter.collect::<Vec<[f32; 4]>>()
                } else {
                    log::debug!("    - No tangents found, using defaults");
                    vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]
                };

                // Extract colors (default if missing)
                let colors = if let Some(iter) = reader.read_colors(0) {
                    iter.into_rgba_f32().collect::<Vec<[f32; 4]>>()
                } else {
                    log::debug!("    - No colors found, using defaults");
                    vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
                };

                // Extract indices (generate a simple triangle list if missing)
                let indices = if let Some(indices) = reader.read_indices() {
                    indices.into_u32().collect::<Vec<u32>>()
                } else {
                    log::debug!("    - No indices found, generating sequential indices");
                    (0..positions.len() as u32).collect()
                };

                log::debug!("    - Found {} indices", indices.len());

                // Validate data lengths
                if tex_coords.len() != positions.len() {
                    log::warn!(
                        "Mismatched vertex attributes: positions ({}), tex_coords ({})",
                        positions.len(),
                        tex_coords.len()
                    );
                }

                if normals.len() != positions.len() {
                    log::warn!(
                        "Mismatched vertex attributes: positions ({}), normals ({})",
                        positions.len(),
                        normals.len()
                    );
                }

                // Create vertices from extracted data
                // Create vertices from extracted data
                let vertices = positions
                    .into_iter()
                    .zip(normals.into_iter())
                    .zip(tex_coords.into_iter())
                    .zip(tangents.into_iter())
                    .zip(colors.into_iter())
                    .map(
                        |((((position, normal), tex_coord), tangent), color)| Vertex {
                            position,
                            normal,
                            uv: tex_coord,
                            tangent,
                            color,
                        },
                    )
                    .collect::<Vec<Vertex>>();

                log::debug!("    - Created {} vertices", vertices.len());

                // Create and add the mesh
                meshes.push(Mesh::new(
                    mesh.name().map(|s| s.to_string()),
                    primitive_type,
                    vertices,
                    indices,
                    primitive.material().index(),
                ));
            }
        }

        // Create nodes with hierarchy
        let nodes: Vec<Node> = gltf
            .nodes()
            .map(|node| {
                // Get the transform matrix and decompose it
                let transform_matrix = node.transform();
                let (translation, rotation, scale) = transform_matrix.decomposed();

                Node {
                    name: node.name().map(|s| s.to_string()),
                    transform: Transform {
                        translation,
                        rotation,
                        scale,
                    },
                    mesh_indices: node.mesh().map(|m| m.index()).into_iter().collect(),
                    children: node.children().map(|c| c.index()).collect(),
                }
            })
            .collect();

        // Find the scene root nodes (currently unused but kept for future use)
        let _scene_root = gltf
            .default_scene()
            .and_then(|scene| scene.nodes().next())
            .map(|node| node.index());

        // Load textures
        let mut textures = Vec::new();

        for (i, texture) in gltf.textures().enumerate() {
            log::debug!(
                "Loading texture {}: {}",
                i,
                texture.name().unwrap_or("unnamed")
            );

            // In a real implementation, we would load the actual texture data here
            // For now, we'll just create a placeholder
            textures.push(Texture::new(
                texture.name().map(|s| s.to_string()),
                1,
                1, // Placeholder size
                TextureFormat::R8G8B8A8Srgb,
                vec![255, 255, 255, 255], // White pixel
                TextureLoadConfig::default(),
            ));
        }

        // Create the loaded model
        let mut model = LoadedModel {
            meshes,
            materials,
            nodes,
            textures,
            scene_root: None, // Will be set below
        };

        // Set the scene root if available
        if let Some(scene) = gltf.default_scene() {
            if let Some(root_node) = scene.nodes().next() {
                model.scene_root = Some(root_node.index());
            }
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PrimitiveType;
    use crate::renderer::Vertex;
    use std::fs;

    // Create a simple test mesh (a quad)
    fn create_test_mesh() -> Mesh {
        let vertices = vec![
            Vertex {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                color: [1.0, 1.0, 1.0, 1.0],
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
    fn test_load_glb() {
        // Test loading a GLB model from the assets directory
        let loader = ModelLoader::new();
        let path = "assets/demon_knight.glb";

        // Skip test if file doesn't exist (asset may not be distributed)
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test_load_glb: {path} not found");
            return;
        }

        // Read the GLB file
        let data = fs::read(path).unwrap_or_else(|_| panic!("Failed to read GLB file at {path}"));

        // Try to load the model
        let result = loader.load_glb(&data);

        // Check if loading was successful
        assert!(result.is_ok(), "Failed to load GLB: {:?}", result.err());

        // Verify the loaded model has the expected structure
        let model = result.unwrap();
        assert!(
            !model.meshes.is_empty(),
            "Model should have at least one mesh"
        );
        assert!(
            !model.materials.is_empty(),
            "Model should have at least one material"
        );
    }

    #[test]
    fn test_generate_lods() {
        // Create a test mesh
        let base_mesh = create_test_mesh();

        // Create a model loader with a mock simplifier
        struct MockSimplifier;

        impl MeshSimplifier for MockSimplifier {
            fn simplify(&self, mesh: &Mesh, _target_vertex_count: usize) -> Mesh {
                // Just return a simplified version (in a real test, this would do actual simplification)
                let simplified = mesh.clone();
                // Just return a triangle (simplest possible mesh)
                let simplified_clone = simplified.clone();
                {
                    let mut mesh_data = simplified_clone.vertices_mut();
                    mesh_data.indices = vec![0, 1, 2];
                }
                simplified_clone
            }

            fn box_clone(&self) -> Box<dyn MeshSimplifier + Send + Sync> {
                Box::new(MockSimplifier)
            }
        }

        let loader = ModelLoader::with_simplifier(
            Arc::new(RwLock::new(AssetMemoryPool::new())),
            MockSimplifier,
        );

        // Generate LODs
        let lods = loader.generate_lods(&base_mesh, 3, 0.5);

        // Verify the LOD model
        assert_eq!(lods.lod_levels.len(), 3, "Should have 3 LOD levels");
        assert_eq!(
            lods.transition_thresholds.len(),
            2,
            "Should have 2 transition thresholds"
        );

        // Test LOD selection
        let high_detail = lods.get_lod(1.0); // Screen size 100%
        let mid_detail = lods.get_lod(0.6); // Screen size 60%
        let low_detail = lods.get_lod(0.1); // Screen size 10%

        // Should get different LODs for different screen sizes
        assert!(Arc::ptr_eq(high_detail, &lods.lod_levels[0]));
        assert!(Arc::ptr_eq(mid_detail, &lods.lod_levels[1]));
        assert!(Arc::ptr_eq(low_detail, &lods.lod_levels[2]));
    }
}
