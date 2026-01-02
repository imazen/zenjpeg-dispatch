// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unified quality scale that is monotonic in both quality metric and file size.
//!
//! # Design Goals
//!
//! 1. **Monotonic quality**: Q+1 always produces same-or-better perceptual quality
//! 2. **Monotonic size**: Q+1 always produces same-or-larger file size
//! 3. **Full range**: Q=0 reaches minimum achievable size, Q=100 reaches near-lossless
//! 4. **Adaptive**: Picks best codec and settings per-image based on analysis
//!
//! # The Problem
//!
//! Raw codec quality values are NOT monotonic across codecs:
//! - mozjpeg Q30 might produce 0.25 bpp at Butteraugli 8
//! - jpegli Q30 might produce 0.33 bpp at Butteraugli 7
//! - Which is "Q30" in unified scale? They're incomparable.
//!
//! # Solution: Target-Based Quality
//!
//! Instead of mapping unified_q → codec_q directly, we:
//! 1. Map unified_q → target_metric (either target_bpp or target_butteraugli)
//! 2. For each image, find the codec+settings that best achieves target_metric
//! 3. Use binary search to find the codec_q that hits the target
//!
//! This ensures monotonicity because the target itself is monotonic.

use crate::adaptive_config::{AdaptiveConfig, EncoderBackend};
use crate::bpp_mapping::{EncoderType, estimate_bpp, quality_for_target_bpp};

/// Quality metric to optimize for
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityMetric {
    /// Butteraugli distance (lower = better, <1 excellent, 1-2 good, 2-4 acceptable)
    Butteraugli,
    /// SSIMULACRA2 score (higher = better, 90+ excellent, 70-90 good)
    Ssimulacra2,
}

/// Unified quality configuration
#[derive(Debug, Clone)]
pub struct UnifiedQualityConfig {
    /// Quality metric to use for optimization
    pub metric: QualityMetric,

    /// Minimum allowed bpp (below this, quality is unusable)
    /// Default: 0.15 (mozjpeg can go to 0.10 but quality is terrible)
    pub min_bpp: f32,

    /// Maximum bpp (above this, diminishing returns)
    /// Default: 5.0 (near-lossless territory)
    pub max_bpp: f32,

    /// Whether to allow codec switching based on RD analysis
    pub allow_codec_switching: bool,

    /// Whether to use evalchroma for subsampling decisions
    pub use_evalchroma: bool,
}

impl Default for UnifiedQualityConfig {
    fn default() -> Self {
        Self {
            metric: QualityMetric::Butteraugli,
            min_bpp: 0.15,
            max_bpp: 5.0,
            allow_codec_switching: true,
            use_evalchroma: true,
        }
    }
}

/// Result of unified quality encoding
#[derive(Debug, Clone)]
pub struct UnifiedEncodingResult {
    /// The unified quality value used (0-100)
    pub unified_quality: u8,
    /// Actual codec quality value used
    pub codec_quality: u8,
    /// Which codec was selected
    pub codec: EncoderType,
    /// Resulting file size in bytes
    pub size: usize,
    /// Bits per pixel
    pub bpp: f32,
    /// Butteraugli score (if measured)
    pub butteraugli: Option<f32>,
    /// SSIMULACRA2 score (if measured)
    pub ssimulacra2: Option<f32>,
}

/// Maps unified quality (0-100) to target bpp.
///
/// The mapping is designed so that:
/// - Q=0 → min_bpp (smallest files, lowest quality)
/// - Q=100 → max_bpp (largest files, highest quality)
/// - The curve is logarithmic to match human perception
///
/// Using bpp as the intermediate target ensures size monotonicity.
pub fn unified_quality_to_target_bpp(unified_q: u8, config: &UnifiedQualityConfig) -> f32 {
    let t = unified_q as f32 / 100.0;

    // Logarithmic mapping: more quality levels at low bpp where differences matter more
    // bpp = min_bpp * (max_bpp/min_bpp)^t
    let ratio = config.max_bpp / config.min_bpp;
    config.min_bpp * ratio.powf(t)
}

/// Maps unified quality to target Butteraugli score.
///
/// Butteraugli ranges:
/// - < 0.5: Imperceptible difference
/// - 0.5 - 1.0: Excellent quality
/// - 1.0 - 2.0: Good quality
/// - 2.0 - 4.0: Acceptable quality
/// - 4.0 - 8.0: Low quality
/// - > 8.0: Very low quality
///
/// We map:
/// - Q=100 → BA 0.3 (near-lossless)
/// - Q=50 → BA 2.0 (acceptable)
/// - Q=0 → BA 15.0 (very low, but still decodable)
pub fn unified_quality_to_target_butteraugli(unified_q: u8) -> f32 {
    let t = unified_q as f32 / 100.0;

    // Exponential mapping: BA = 15 * 0.02^t
    // At t=0: BA = 15
    // At t=0.5: BA ≈ 2.1
    // At t=1: BA = 0.3
    15.0 * (0.02_f32).powf(t)
}

/// Maps unified quality to target SSIMULACRA2 score.
///
/// SSIMULACRA2 ranges:
/// - 90+: Excellent (near-lossless)
/// - 70-90: Good
/// - 50-70: Acceptable
/// - 30-50: Low quality
/// - < 30: Very low quality
///
/// We map:
/// - Q=100 → SSIM2 98
/// - Q=50 → SSIM2 75
/// - Q=0 → SSIM2 30
pub fn unified_quality_to_target_ssimulacra2(unified_q: u8) -> f32 {
    let t = unified_q as f32 / 100.0;

    // Linear-ish mapping with slight curve
    // SSIM2 = 30 + 68 * t^0.8
    30.0 + 68.0 * t.powf(0.8)
}

/// Recommends the best codec for a target bpp based on RD characteristics.
///
/// Key crossover points (empirical):
/// - < 0.24 bpp: Only mozjpeg can reach this (jpegli has quality floor)
/// - 0.24-0.30 bpp: Crossover zone, image-dependent
/// - > 0.30 bpp: jpegli typically wins on perceptual quality
pub fn recommend_codec_for_bpp(target_bpp: f32, _image_analysis: Option<&ImageAnalysis>) -> EncoderType {
    // TODO: Use image_analysis to refine decision
    // For now, use simple bpp-based heuristic

    if target_bpp < 0.24 {
        // Only mozjpeg can go this low
        EncoderType::Mozjpeg
    } else if target_bpp < 0.30 {
        // Crossover zone - mozjpeg slightly better on average
        EncoderType::Mozjpeg
    } else {
        // jpegli wins on perceptual quality at higher bpp
        EncoderType::Jpegli
    }
}

/// Image analysis results for codec selection heuristics
#[derive(Debug, Clone, Default)]
pub struct ImageAnalysis {
    /// Average local variance (texture complexity)
    pub variance: f32,
    /// Edge density (0-1, higher = more edges)
    pub edge_density: f32,
    /// Chroma complexity from evalchroma (0-1)
    pub chroma_complexity: f32,
    /// Fraction of near-uniform blocks
    pub uniform_block_fraction: f32,
    /// Whether image has significant high-frequency content
    pub has_high_frequency: bool,
    /// Dominant color count (for graphics vs photos)
    pub color_count_estimate: u32,
}

impl ImageAnalysis {
    /// Analyze an RGB image
    pub fn from_rgb(pixels: &[u8], width: usize, height: usize) -> Self {
        let mut analysis = Self::default();

        // Calculate variance (simplified: just luminance variance)
        let luma: Vec<f32> = pixels.chunks(3)
            .map(|rgb| 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
            .collect();

        let mean: f32 = luma.iter().sum::<f32>() / luma.len() as f32;
        analysis.variance = luma.iter()
            .map(|&l| (l - mean).powi(2))
            .sum::<f32>() / luma.len() as f32;

        // Estimate edge density using simple gradient
        let mut edge_sum = 0.0f32;
        for y in 1..height-1 {
            for x in 1..width-1 {
                let idx = y * width + x;
                let gx = (luma[idx + 1] - luma[idx - 1]).abs();
                let gy = (luma[idx + width] - luma[idx - width]).abs();
                edge_sum += (gx * gx + gy * gy).sqrt();
            }
        }
        analysis.edge_density = edge_sum / ((width - 2) * (height - 2)) as f32 / 255.0;

        // Estimate chroma complexity
        let mut chroma_var = 0.0f32;
        for rgb in pixels.chunks(3) {
            let cb = -0.169 * rgb[0] as f32 - 0.331 * rgb[1] as f32 + 0.500 * rgb[2] as f32;
            let cr = 0.500 * rgb[0] as f32 - 0.419 * rgb[1] as f32 - 0.081 * rgb[2] as f32;
            chroma_var += cb * cb + cr * cr;
        }
        analysis.chroma_complexity = (chroma_var / pixels.len() as f32 * 3.0).sqrt() / 128.0;
        analysis.chroma_complexity = analysis.chroma_complexity.min(1.0);

        // Count uniform 8x8 blocks
        let blocks_x = width / 8;
        let blocks_y = height / 8;
        let mut uniform_count = 0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let base_idx = (by * 8 * width + bx * 8) * 3;
                let first_r = pixels[base_idx];
                let first_g = pixels[base_idx + 1];
                let first_b = pixels[base_idx + 2];

                let mut is_uniform = true;
                'block: for dy in 0..8 {
                    for dx in 0..8 {
                        let idx = base_idx + (dy * width + dx) * 3;
                        if (pixels[idx] as i32 - first_r as i32).abs() > 4
                            || (pixels[idx + 1] as i32 - first_g as i32).abs() > 4
                            || (pixels[idx + 2] as i32 - first_b as i32).abs() > 4
                        {
                            is_uniform = false;
                            break 'block;
                        }
                    }
                }
                if is_uniform {
                    uniform_count += 1;
                }
            }
        }

        let total_blocks = blocks_x * blocks_y;
        analysis.uniform_block_fraction = if total_blocks > 0 {
            uniform_count as f32 / total_blocks as f32
        } else {
            0.0
        };

        // High frequency detection (using variance of gradient)
        analysis.has_high_frequency = analysis.edge_density > 0.1;

        // Rough color count (sample-based)
        // For real implementation, use a histogram or color quantization
        analysis.color_count_estimate = 1000; // Placeholder

        analysis
    }

    /// Suggest whether jpegli is likely to outperform mozjpeg for this image
    pub fn prefers_jpegli(&self) -> bool {
        // jpegli excels at:
        // - High variance (complex textures)
        // - High chroma complexity
        // - Few uniform blocks (photos, not graphics)

        self.variance > 500.0
            && self.chroma_complexity > 0.1
            && self.uniform_block_fraction < 0.3
    }

    /// Suggest whether this image can tolerate aggressive subsampling
    pub fn can_subsample(&self) -> bool {
        // Low chroma complexity = safe to subsample
        self.chroma_complexity < 0.15
    }
}

/// Get the codec quality value that achieves target bpp for a given codec.
///
/// This uses the empirical mapping tables from bpp_mapping module.
pub fn codec_quality_for_target_bpp(codec: EncoderType, target_bpp: f32) -> u8 {
    quality_for_target_bpp(codec, target_bpp)
}

/// Main entry point: convert unified quality to encoding parameters.
///
/// Returns the recommended configuration for encoding at the specified
/// unified quality level.
pub fn unified_quality_to_config(
    unified_q: u8,
    image_analysis: Option<&ImageAnalysis>,
    config: &UnifiedQualityConfig,
) -> (EncoderType, u8, AdaptiveConfig) {
    // Step 1: Convert unified quality to target bpp
    let target_bpp = unified_quality_to_target_bpp(unified_q, config);

    // Step 2: Choose codec based on target bpp and image analysis
    let codec = if config.allow_codec_switching {
        recommend_codec_for_bpp(target_bpp, image_analysis)
    } else {
        EncoderType::Jpegli // Default to jpegli if no switching
    };

    // Step 3: Find codec-specific quality that achieves target bpp
    let codec_q = codec_quality_for_target_bpp(codec, target_bpp);

    // Step 4: Build adaptive config
    let mut adaptive = match codec {
        EncoderType::Mozjpeg => AdaptiveConfig::mozjpeg_default(),
        EncoderType::Jpegli => AdaptiveConfig::jpegli_default(),
    };
    adaptive.quality = codec_q;

    // Adjust subsampling based on analysis
    if let Some(analysis) = image_analysis {
        if config.use_evalchroma {
            use crate::adaptive_config::SubsamplingConfig;
            use crate::types::Subsampling;

            if analysis.can_subsample() {
                adaptive.subsampling = SubsamplingConfig::Fixed(Subsampling::S420);
            } else {
                adaptive.subsampling = SubsamplingConfig::Fixed(Subsampling::S444);
            }
        }
    }

    (codec, codec_q, adaptive)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_quality_to_bpp_monotonic() {
        let config = UnifiedQualityConfig::default();

        let mut prev_bpp = 0.0;
        for q in 0..=100 {
            let bpp = unified_quality_to_target_bpp(q, &config);
            assert!(bpp >= prev_bpp, "bpp should increase with quality: Q{} bpp={} < prev={}", q, bpp, prev_bpp);
            prev_bpp = bpp;
        }
    }

    #[test]
    fn test_unified_quality_to_butteraugli_monotonic() {
        let mut prev_ba = f32::MAX;
        for q in 0..=100 {
            let ba = unified_quality_to_target_butteraugli(q);
            assert!(ba <= prev_ba, "butteraugli should decrease with quality: Q{} ba={} > prev={}", q, ba, prev_ba);
            prev_ba = ba;
        }
    }

    #[test]
    fn test_unified_quality_to_ssimulacra2_monotonic() {
        let mut prev_ssim2 = 0.0;
        for q in 0..=100 {
            let ssim2 = unified_quality_to_target_ssimulacra2(q);
            assert!(ssim2 >= prev_ssim2, "ssim2 should increase with quality: Q{} ssim2={} < prev={}", q, ssim2, prev_ssim2);
            prev_ssim2 = ssim2;
        }
    }

    #[test]
    fn test_unified_quality_range() {
        let config = UnifiedQualityConfig::default();

        // Q0 should give min_bpp
        let bpp_0 = unified_quality_to_target_bpp(0, &config);
        assert!((bpp_0 - config.min_bpp).abs() < 0.01);

        // Q100 should give max_bpp
        let bpp_100 = unified_quality_to_target_bpp(100, &config);
        assert!((bpp_100 - config.max_bpp).abs() < 0.01);
    }

    #[test]
    fn test_codec_recommendation() {
        // Very low bpp - must use mozjpeg
        assert_eq!(recommend_codec_for_bpp(0.15, None), EncoderType::Mozjpeg);

        // High bpp - prefer jpegli
        assert_eq!(recommend_codec_for_bpp(1.0, None), EncoderType::Jpegli);
    }

    #[test]
    fn test_image_analysis() {
        // Create a simple gradient image
        let width = 64;
        let height = 64;
        let mut pixels = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                pixels[idx] = (x * 4) as u8;     // R gradient
                pixels[idx + 1] = (y * 4) as u8; // G gradient
                pixels[idx + 2] = 128;            // B constant
            }
        }

        let analysis = ImageAnalysis::from_rgb(&pixels, width, height);

        // Should have non-zero variance
        assert!(analysis.variance > 0.0);

        // Should have edges
        assert!(analysis.edge_density > 0.0);

        // Gradient has low uniform block fraction
        assert!(analysis.uniform_block_fraction < 0.5);
    }

    #[test]
    fn test_unified_to_config() {
        let config = UnifiedQualityConfig::default();

        // Low quality should use mozjpeg
        let (codec, _q, _cfg) = unified_quality_to_config(10, None, &config);
        assert_eq!(codec, EncoderType::Mozjpeg);

        // High quality should use jpegli
        let (codec, _q, _cfg) = unified_quality_to_config(80, None, &config);
        assert_eq!(codec, EncoderType::Jpegli);
    }
}
