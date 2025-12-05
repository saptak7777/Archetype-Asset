//! Vulkan GPU implementation
//!
//! Provides a Vulkan backend using `ash` and `vk-mem`.

use super::{BufferUsage, GpuDevice, GpuError, GpuResult, GpuTextureFormat, TextureDescriptor};
use ash::vk;
use parking_lot::Mutex;
use std::fmt;
use std::sync::Arc;
use vk_mem::Alloc;

/// Inner state for VulkanBuffer that holds the actual Vulkan resources
struct VulkanBufferInner {
    /// Vulkan buffer handle
    buffer: vk::Buffer,
    /// VMA allocation
    allocation: vk_mem::Allocation,
    /// Buffer size in bytes
    size: usize,
    /// Buffer usage
    #[allow(dead_code)]
    usage: BufferUsage,
    /// Reference to allocator for cleanup
    allocator: Arc<Mutex<vk_mem::Allocator>>,
}

impl Drop for VulkanBufferInner {
    fn drop(&mut self) {
        let allocator = self.allocator.lock();
        unsafe {
            allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
        log::trace!("VulkanBuffer destroyed (size={})", self.size);
    }
}

/// Vulkan buffer with associated allocation
#[derive(Clone)]
pub struct VulkanBuffer {
    inner: Arc<Mutex<VulkanBufferInner>>,
    /// Vulkan buffer handle (cached for quick access)
    pub buffer: vk::Buffer,
    /// Buffer size in bytes (cached for quick access)
    pub size: usize,
    /// Buffer usage (cached for quick access)
    pub usage: BufferUsage,
}

impl fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("size", &self.size)
            .field("usage", &self.usage)
            .finish()
    }
}

impl VulkanBuffer {
    /// Get the raw Vulkan buffer handle
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }
}

/// Inner state for VulkanTexture that holds the actual Vulkan resources
struct VulkanTextureInner {
    /// Vulkan image handle
    image: vk::Image,
    /// Image view
    view: vk::ImageView,
    /// VMA allocation
    allocation: vk_mem::Allocation,
    /// Reference to allocator for cleanup
    allocator: Arc<Mutex<vk_mem::Allocator>>,
    /// Reference to device for cleanup
    device: ash::Device,
}

impl Drop for VulkanTextureInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
        }
        let allocator = self.allocator.lock();
        unsafe {
            allocator.destroy_image(self.image, &mut self.allocation);
        }
        log::trace!("VulkanTexture destroyed");
    }
}

/// Vulkan texture with associated image and view
#[derive(Clone)]
pub struct VulkanTexture {
    #[allow(dead_code)]
    inner: Arc<Mutex<VulkanTextureInner>>,
    /// Vulkan image handle (cached for quick access)
    pub image: vk::Image,
    /// Image view (cached for quick access)
    pub view: vk::ImageView,
    /// Texture width (cached for quick access)
    pub width: u32,
    /// Texture height (cached for quick access)
    pub height: u32,
    /// Format (cached for quick access)
    pub format: GpuTextureFormat,
}

impl fmt::Debug for VulkanTexture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanTexture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .finish()
    }
}

impl VulkanTexture {
    /// Get the raw Vulkan image handle
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Get the image view handle
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
}

/// Vulkan GPU device wrapper
///
/// Wraps `ash::Device` and VMA allocator to implement the `GpuDevice` trait.
#[derive(Clone)]
pub struct VulkanDevice {
    device: ash::Device,
    allocator: Arc<Mutex<vk_mem::Allocator>>,
}

impl fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("backend", &"Vulkan")
            .finish()
    }
}

impl VulkanDevice {
    /// Create a new Vulkan device wrapper
    ///
    /// # Arguments
    /// * `device` - The ash Vulkan device
    /// * `allocator` - VMA allocator for memory management
    pub fn new(device: ash::Device, allocator: Arc<Mutex<vk_mem::Allocator>>) -> Self {
        Self { device, allocator }
    }

    /// Get the underlying ash device
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Get the allocator
    pub fn allocator(&self) -> &Arc<Mutex<vk_mem::Allocator>> {
        &self.allocator
    }

    /// Convert BufferUsage to Vulkan buffer usage flags
    fn to_vk_buffer_usage(usage: BufferUsage) -> vk::BufferUsageFlags {
        match usage {
            BufferUsage::Vertex => {
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Index => {
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Uniform => {
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Storage => {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
            }
            BufferUsage::Staging => vk::BufferUsageFlags::TRANSFER_SRC,
        }
    }

    /// Convert GpuTextureFormat to Vulkan format
    fn to_vk_format(format: GpuTextureFormat) -> vk::Format {
        match format {
            GpuTextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            GpuTextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            GpuTextureFormat::Rgb8Srgb => vk::Format::R8G8B8_SRGB,
            GpuTextureFormat::R8Unorm => vk::Format::R8_UNORM,
            GpuTextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
            GpuTextureFormat::Depth24Stencil8 => vk::Format::D24_UNORM_S8_UINT,
        }
    }
}

impl GpuDevice for VulkanDevice {
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;

    fn allocate_buffer(&self, size: usize, usage: BufferUsage) -> GpuResult<Self::Buffer> {
        if size == 0 {
            return Err(GpuError::InvalidSize(size));
        }

        let vk_usage = Self::to_vk_buffer_usage(usage);

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as u64)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        // Determine memory location based on usage
        let memory_location = match usage {
            BufferUsage::Staging => vk_mem::MemoryUsage::AutoPreferHost,
            _ => vk_mem::MemoryUsage::AutoPreferDevice,
        };

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: memory_location,
            ..Default::default()
        };

        let allocator = self.allocator.lock();
        let (buffer, allocation) = unsafe {
            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .map_err(|e| GpuError::AllocationFailed(format!("VMA error: {e:?}")))?
        };

        let inner = VulkanBufferInner {
            buffer,
            allocation,
            size,
            usage,
            allocator: Arc::clone(&self.allocator),
        };

        Ok(VulkanBuffer {
            inner: Arc::new(Mutex::new(inner)),
            buffer,
            size,
            usage,
        })
    }

    fn upload_buffer_data(
        &self,
        buffer: &Self::Buffer,
        offset: usize,
        data: &[u8],
    ) -> GpuResult<()> {
        if offset + data.len() > buffer.size {
            return Err(GpuError::UploadFailed(format!(
                "Data exceeds buffer size: offset={}, data_len={}, buffer_size={}",
                offset,
                data.len(),
                buffer.size
            )));
        }

        let mut inner = buffer.inner.lock();
        let allocator = self.allocator.lock();

        // Map memory, copy data, unmap
        let ptr = unsafe {
            allocator
                .map_memory(&mut inner.allocation)
                .map_err(|e| GpuError::UploadFailed(format!("Failed to map memory: {e:?}")))?
        };

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
            allocator.unmap_memory(&mut inner.allocation);
        }

        Ok(())
    }

    fn create_texture(&self, desc: &TextureDescriptor, data: &[u8]) -> GpuResult<Self::Texture> {
        if desc.width == 0 || desc.height == 0 {
            return Err(GpuError::TextureCreationFailed(
                "Invalid texture dimensions".to_string(),
            ));
        }

        let vk_format = Self::to_vk_format(desc.format);

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .build();

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let allocator = self.allocator.lock();
        let (image, allocation) = unsafe {
            allocator
                .create_image(&image_info, &allocation_info)
                .map_err(|e| GpuError::TextureCreationFailed(format!("VMA error: {e:?}")))?
        };

        // Create image view
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: desc.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        let view = unsafe { self.device.create_image_view(&view_info, None) }.map_err(|e| {
            GpuError::TextureCreationFailed(format!("Failed to create image view: {e:?}"))
        })?;

        // Note: Actual data upload would require staging buffer + command buffer
        // This is a simplified implementation
        let _ = data; // Data upload happens separately via staging buffer

        let inner = VulkanTextureInner {
            image,
            view,
            allocation,
            allocator: Arc::clone(&self.allocator),
            device: self.device.clone(),
        };

        Ok(VulkanTexture {
            inner: Arc::new(Mutex::new(inner)),
            image,
            view,
            width: desc.width,
            height: desc.height,
            format: desc.format,
        })
    }

    fn destroy_buffer(&self, _buffer: Self::Buffer) {
        // Drop is handled by VulkanBufferInner
    }

    fn destroy_texture(&self, _texture: Self::Texture) {
        // Drop is handled by VulkanTextureInner
    }

    fn backend_name(&self) -> &'static str {
        "Vulkan"
    }
}
