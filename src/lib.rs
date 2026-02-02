//! # zenjpeg-dispatch - JPEG Encoder Dispatcher
//!
//! zenjpeg-dispatch intelligently selects between mozjpeg and jpegli encoders
//! based on image characteristics to achieve Pareto-optimal compression.
//!
//! ## Key Features
//!
//! - **Adaptive codec selection**: Automatically chooses the best encoder
//!   (mozjpeg or jpegli) based on target quality and image content
//! - **Trellis quantization** (via mozjpeg): Rate-distortion optimized
//!   coefficient selection, especially effective at low quality
//! - **Adaptive quantization** (via jpegli): Content-aware bit allocation
//!   based on perceptual importance
//! - **Perceptual optimization**: Uses Butteraugli/SSIMULACRA2 for quality
//!   assessment and tuning
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zenjpeg_dispatch::{Encoder, Quality};
//!
//! let encoder = Encoder::new()
//!     .quality(Quality::Perceptual(85.0))
//!     .optimize_for_web(true);
//!
//! let jpeg_data = encoder.encode_rgb(&pixels, width, height)?;
//! ```
//!
//! ## Quality Modes
//!
//! - `Quality::Low(q)`: Uses mozjpeg-style trellis for best low-bitrate results
//! - `Quality::High(q)`: Uses jpegli-style adaptive quantization for high quality
//! - `Quality::Perceptual(target)`: Automatically selects strategy to hit target
//!   perceptual quality (SSIMULACRA2 score)

// Core modules
pub mod analysis;
mod consts;
mod consts_moz;
mod error;
mod types;

// Encoding pipeline
mod color;
mod dct;
mod encode;
mod entropy;
mod huffman;
mod quant;

// Advanced features
pub mod adaptive_quant;
mod deringing;
mod progressive;
pub mod sa_tables;
mod trellis;

// Strategy selection
mod strategy;

// BPP and SSIM2 mapping tables
pub mod bpp_mapping;

// Runtime-configurable adaptive features
pub mod adaptive_config;

// Unified monotonic quality scale
pub mod unified_quality;

// Note: jpegli encoding is now delegated to the jpegli-rs crate dependency

// Public API
pub use encode::Encoder;
pub use error::Error;
pub use progressive::ScanInfo;
pub use types::{
    ColorSpace, EncodingStrategy, OptimizeFor, PixelFormat, Quality, ScanScript, Subsampling,
};

// Analysis and codec selection
pub use analysis::{
    analyze_image, select_codec_auto, select_codec_for_butteraugli, select_codec_for_dssim,
    CodecRecommendation, ImageAnalysis, RecommendedApproach,
};

// Unified quality scale (Z = 0-100, where Z ≈ SSIM2)
pub use unified_quality::{
    butteraugli_to_unified_quality, dssim_to_unified_quality, select_codec_for_z,
    ssimulacra2_to_unified_quality, unified_quality_to_target_butteraugli,
    unified_quality_to_target_dssim, unified_quality_to_target_ssimulacra2,
};

/// Result type for zenjpeg-dispatch operations
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode() {
        // Placeholder test - will be implemented with actual encoder
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn debug_quality_encoding() {
        use quant::QuantTableSet;

        // Check quant table values at different quality levels
        for q in [30, 50, 70, 95] {
            let tables = QuantTableSet::standard(q as u8);
            println!(
                "Q{}: luma[0]={}, luma[1]={}, luma[63]={}",
                q, tables.luma.values[0], tables.luma.values[1], tables.luma.values[63]
            );
        }
    }
}
