//! Texture loading and processing

use image::io::Reader as ImageReader;
use image::ImageFormat;
use std::path::Path;
use thiserror::Error;

/// Error type for texture loading operations
#[derive(Error, Debug)]
pub enum TextureError {
    #[error("Image decoding error: {0}")]
    DecodeError(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Represents a loaded texture
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: TextureFormat,
}

/// Supported texture formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextureFormat {
    Rgba8,
    Rgb8,
}

/// Loads and processes texture files
#[derive(Default, Clone)]
pub struct TextureLoader;

impl TextureLoader {
    /// Create a new texture loader
    pub fn new() -> Self {
        Self
    }

    /// Load a texture from binary data
    pub fn load(&self, data: &[u8]) -> Result<Texture, TextureError> {
        let format =
            image::guess_format(data).map_err(|e| TextureError::DecodeError(e.to_string()))?;

        match format {
            ImageFormat::Jpeg | ImageFormat::Png => {}
            _ => {
                return Err(TextureError::UnsupportedFormat(format!(
                    "Only JPG/JPEG and PNG formats are supported, got {:?}",
                    format.extensions_str()
                )))
            }
        }

        let img = ImageReader::with_format(std::io::Cursor::new(data), format)
            .decode()
            .map_err(|e| TextureError::DecodeError(e.to_string()))?;

        let rgba_img = img.into_rgba8();
        let (width, height) = rgba_img.dimensions();

        Ok(Texture {
            width,
            height,
            data: rgba_img.into_raw(),
            format: TextureFormat::Rgba8,
        })
    }

    /// Load a texture from a file synchronously
    pub fn load_file_sync<P: AsRef<Path>>(&self, path: P) -> Result<Texture, TextureError> {
        let data = std::fs::read(path)?;
        self.load(&data)
    }
}

// Async loading (requires tokio)
#[cfg(feature = "runtime-tokio")]
impl TextureLoader {
    /// Load a texture from a file asynchronously
    pub async fn load_file<P: AsRef<Path>>(&self, path: P) -> Result<Texture, TextureError> {
        let data = tokio::fs::read(path).await?;
        self.load(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_jpeg_sync() {
        let loader = TextureLoader::new();

        let mut img = image::RgbaImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));

        let mut jpeg_data = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut jpeg_data),
            image::ImageFormat::Jpeg,
        )
        .expect("Failed to encode test image");

        let result = loader.load(&jpeg_data);
        assert!(result.is_ok(), "Failed to load JPEG: {:?}", result.err());

        let texture = result.unwrap();
        assert_eq!(texture.width, 1);
        assert_eq!(texture.height, 1);
        assert_eq!(texture.format, TextureFormat::Rgba8);
    }

    #[test]
    fn test_load_png_sync() {
        let loader = TextureLoader::new();

        let mut img = image::RgbaImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));

        let mut png_data = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut png_data),
            image::ImageFormat::Png,
        )
        .expect("Failed to encode test image");

        let result = loader.load(&png_data);
        assert!(result.is_ok(), "Failed to load PNG: {:?}", result.err());

        let texture = result.unwrap();
        assert_eq!(texture.width, 1);
        assert_eq!(texture.height, 1);
    }
}
