//! Mock GPU implementation for testing
//!
//! Provides a mock GPU that stores data in memory for testing
//! without requiring actual GPU hardware.

use super::{BufferUsage, GpuDevice, GpuError, GpuResult, GpuTextureFormat, TextureDescriptor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Counter for generating unique buffer/texture IDs
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Mock GPU device for testing
///
/// This implementation stores all data in memory and can be used
/// for unit tests without requiring actual GPU hardware.
#[derive(Clone, Debug, Default)]
pub struct MockGpu {
    /// Track total allocated memory for testing
    #[allow(dead_code)]
    allocated_bytes: Arc<AtomicU64>,
}

impl MockGpu {
    /// Create a new mock GPU device
    pub fn new() -> Self {
        Self {
            allocated_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get total allocated memory (for testing)
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }
}

/// Mock buffer that stores data in memory
#[derive(Clone, Debug)]
pub struct MockBuffer {
    /// Unique identifier
    pub id: u64,
    /// Buffer data
    pub data: Arc<parking_lot::RwLock<Vec<u8>>>,
    /// Buffer usage
    pub usage: BufferUsage,
}

impl MockBuffer {
    /// Create a new mock buffer
    pub fn new(size: usize, usage: BufferUsage) -> Self {
        Self {
            id: next_id(),
            data: Arc::new(parking_lot::RwLock::new(vec![0u8; size])),
            usage,
        }
    }

    /// Get the size of the buffer
    pub fn size(&self) -> usize {
        self.data.read().len()
    }

    /// Read buffer data
    pub fn read_data(&self) -> Vec<u8> {
        self.data.read().clone()
    }
}

/// Mock texture that stores pixel data in memory
#[derive(Clone, Debug)]
pub struct MockTexture {
    /// Unique identifier
    pub id: u64,
    /// Texture width
    pub width: u32,
    /// Texture height
    pub height: u32,
    /// Texture format
    pub format: GpuTextureFormat,
    /// Pixel data
    pub data: Arc<parking_lot::RwLock<Vec<u8>>>,
    /// Number of mip levels
    pub mip_levels: u32,
}

impl MockTexture {
    /// Create a new mock texture
    pub fn new(desc: &TextureDescriptor, data: &[u8]) -> Self {
        Self {
            id: next_id(),
            width: desc.width,
            height: desc.height,
            format: desc.format,
            data: Arc::new(parking_lot::RwLock::new(data.to_vec())),
            mip_levels: desc.mip_levels,
        }
    }
}

impl GpuDevice for MockGpu {
    type Buffer = MockBuffer;
    type Texture = MockTexture;

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> GpuResult<Self::Buffer> {
        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        self.allocated_bytes
            .fetch_add(size as u64, Ordering::Relaxed);
        Ok(MockBuffer::new(size, usage))
    }

    fn upload_buffer_data(
        &self,
        buffer: &Self::Buffer,
        offset: usize,
        data: &[u8],
    ) -> GpuResult<()> {
        let mut buf_data = buffer.data.write();

        if offset + data.len() > buf_data.len() {
            return Err(GpuError::UploadFailed(format!(
                "Data exceeds buffer size: offset={}, data_len={}, buffer_size={}",
                offset,
                data.len(),
                buf_data.len()
            )));
        }

        buf_data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn create_texture(&self, desc: &TextureDescriptor, data: &[u8]) -> GpuResult<Self::Texture> {
        if desc.width == 0 || desc.height == 0 {
            return Err(GpuError::TextureCreationFailed(
                "Invalid texture dimensions".to_string(),
            ));
        }

        let expected_size = (desc.width * desc.height * 4) as usize; // Assume RGBA
        self.allocated_bytes
            .fetch_add(expected_size as u64, Ordering::Relaxed);

        Ok(MockTexture::new(desc, data))
    }

    fn destroy_buffer(&self, buffer: Self::Buffer) {
        let size = buffer.size() as u64;
        self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    fn destroy_texture(&self, texture: Self::Texture) {
        let size = (texture.width * texture.height * 4) as u64;
        self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    fn backend_name(&self) -> &'static str {
        "Mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_gpu_allocate_buffer() {
        let gpu = MockGpu::new();
        let buffer = gpu.allocate_buffer(1024, BufferUsage::Vertex).unwrap();

        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.usage, BufferUsage::Vertex);
        assert_eq!(gpu.allocated_bytes(), 1024);
    }

    #[test]
    fn test_mock_gpu_upload_buffer() {
        let gpu = MockGpu::new();
        let buffer = gpu.allocate_buffer(1024, BufferUsage::Vertex).unwrap();

        let data = vec![1u8, 2, 3, 4];
        gpu.upload_buffer_data(&buffer, 0, &data).unwrap();

        let read_data = buffer.read_data();
        assert_eq!(&read_data[0..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_mock_gpu_upload_with_offset() {
        let gpu = MockGpu::new();
        let buffer = gpu.allocate_buffer(1024, BufferUsage::Vertex).unwrap();

        let data = vec![5u8, 6, 7, 8];
        gpu.upload_buffer_data(&buffer, 100, &data).unwrap();

        let read_data = buffer.read_data();
        assert_eq!(&read_data[100..104], &[5, 6, 7, 8]);
    }

    #[test]
    fn test_mock_gpu_upload_overflow() {
        let gpu = MockGpu::new();
        let buffer = gpu.allocate_buffer(10, BufferUsage::Vertex).unwrap();

        let data = vec![0u8; 20];
        let result = gpu.upload_buffer_data(&buffer, 0, &data);

        assert!(result.is_err());
    }

    #[test]
    fn test_mock_gpu_create_texture() {
        let gpu = MockGpu::new();
        let desc = TextureDescriptor {
            width: 64,
            height: 64,
            format: GpuTextureFormat::Rgba8Srgb,
            mip_levels: 1,
            generate_mipmaps: false,
        };

        let data = vec![0u8; 64 * 64 * 4];
        let texture = gpu.create_texture(&desc, &data).unwrap();

        assert_eq!(texture.width, 64);
        assert_eq!(texture.height, 64);
    }

    #[test]
    fn test_mock_gpu_zero_size_buffer() {
        let gpu = MockGpu::new();
        let result = gpu.allocate_buffer(0, BufferUsage::Vertex);

        assert!(result.is_err());
    }

    #[test]
    fn test_mock_gpu_destroy_buffer() {
        let gpu = MockGpu::new();
        let buffer = gpu.allocate_buffer(1024, BufferUsage::Vertex).unwrap();

        assert_eq!(gpu.allocated_bytes(), 1024);
        gpu.destroy_buffer(buffer);
        assert_eq!(gpu.allocated_bytes(), 0);
    }

    #[test]
    fn test_mock_gpu_clone() {
        let gpu1 = MockGpu::new();
        let _buffer = gpu1.allocate_buffer(1024, BufferUsage::Vertex).unwrap();

        let gpu2 = gpu1.clone();
        assert_eq!(gpu2.allocated_bytes(), 1024);
    }
}
