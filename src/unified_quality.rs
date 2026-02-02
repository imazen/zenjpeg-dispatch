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
use crate::bpp_mapping::{estimate_bpp, quality_for_target_bpp, EncoderType};

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

/// Maps unified quality Z (0-100) to target Butteraugli distance.
///
/// **Data-driven mapping** fitted from 18,191 data points across 86 images.
/// Z is designed to equal SSIMULACRA2, so Z=75 means SSIM2=75.
///
/// Formula: BA = 8.942 * exp(-0.01411 * Z)
/// R² = 0.957
///
/// | Z | Butteraugli | Quality Level |
/// |---|-------------|---------------|
/// | 90 | 2.5 | Excellent |
/// | 75 | 3.1 | Good |
/// | 50 | 4.4 | Acceptable |
/// | 25 | 6.3 | Low |
/// | 0 | 8.9 | Very Low |
pub fn unified_quality_to_target_butteraugli(unified_q: u8) -> f32 {
    // Data-driven exponential fit from 18,191 samples
    // BA(Z) = A * exp(-B * Z) + C
    const A: f32 = 8.942026;
    const B: f32 = 0.014111;
    const C: f32 = 0.0;

    let z = unified_q as f32;
    A * (-B * z).exp() + C
}

/// Convert Butteraugli distance to unified quality Z (0-100).
///
/// Inverse of `unified_quality_to_target_butteraugli`.
/// Z = -ln((BA - C) / A) / B
#[must_use]
pub fn butteraugli_to_unified_quality(ba: f32) -> f32 {
    const A: f32 = 8.942026;
    const B: f32 = 0.014111;
    const C: f32 = 0.0;

    if ba <= C {
        return 100.0;
    }
    let ratio = (ba - C) / A;
    if ratio <= 0.0 {
        return 100.0;
    }
    (-ratio.ln() / B).clamp(0.0, 100.0)
}

/// Maps unified quality Z (0-100) to target SSIMULACRA2 score.
///
/// **By design, Z = SSIM2.** This is the anchor metric for the unified scale.
///
/// SSIMULACRA2 ranges:
/// - 90+: Excellent (near-lossless)
/// - 70-90: Good
/// - 50-70: Acceptable
/// - 30-50: Low quality
/// - < 30: Very low quality
pub fn unified_quality_to_target_ssimulacra2(unified_q: u8) -> f32 {
    // Z = SSIM2 by design (direct mapping)
    unified_q as f32
}

/// Convert SSIMULACRA2 score to unified quality Z (0-100).
///
/// By design, Z = SSIM2.
#[must_use]
pub fn ssimulacra2_to_unified_quality(ssim2: f32) -> f32 {
    ssim2.clamp(-10.0, 100.0)
}

/// Maps unified quality Z (0-100) to target DSSIM value.
///
/// **Data-driven mapping** fitted from 18,191 data points across 86 images.
///
/// Formula: DSSIM = 0.02277 * exp(-0.02589 * Z)
/// R² = 0.979
///
/// | Z | DSSIM | Quality Level |
/// |---|-------|---------------|
/// | 90 | 0.0022 | Excellent |
/// | 75 | 0.0033 | Good |
/// | 50 | 0.0062 | Acceptable |
/// | 25 | 0.0117 | Low |
/// | 0 | 0.0228 | Very Low |
#[must_use]
pub fn unified_quality_to_target_dssim(unified_q: u8) -> f32 {
    // Data-driven exponential fit from 18,191 samples
    // DSSIM(Z) = A * exp(-B * Z) + C
    const A: f32 = 0.022_769_09;
    const B: f32 = 0.025_895;
    const C: f32 = 0.0;

    let z = unified_q as f32;
    A * (-B * z).exp() + C
}

/// Convert DSSIM value to unified quality Z (0-100).
///
/// Inverse of `unified_quality_to_target_dssim`.
/// Z = -ln((DSSIM - C) / A) / B
#[must_use]
pub fn dssim_to_unified_quality(dssim: f32) -> f32 {
    const A: f32 = 0.022_769_09;
    const B: f32 = 0.025_895;
    const C: f32 = 0.0;

    if dssim <= C {
        return 100.0;
    }
    let ratio = (dssim - C) / A;
    if ratio <= 0.0 {
        return 100.0;
    }
    (-ratio.ln() / B).clamp(0.0, 100.0)
}

/// Recommends the best codec for a target bpp based on RD characteristics.
///
/// Key crossover points (empirical):
/// - < 0.24 bpp: Only mozjpeg can reach this (jpegli has quality floor)
/// - 0.24-0.30 bpp: Crossover zone, image-dependent
/// - > 0.30 bpp: jpegli typically wins on perceptual quality
pub fn recommend_codec_for_bpp(
    target_bpp: f32,
    _image_analysis: Option<&ImageAnalysis>,
) -> EncoderType {
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

/// Select the optimal codec for a target unified quality Z.
///
/// This combines the metric-specific heuristics, routing to the appropriate
/// selection function based on which metric you're optimizing for.
///
/// # Key Insight
///
/// Different metrics have **opposite** optimal codec preferences:
/// - **Butteraugli**: jpegli wins at most quality levels
/// - **DSSIM**: mozjpeg wins at most quality levels
/// - **SSIMULACRA2**: similar to Butteraugli (perceptual correlation)
///
/// # Arguments
///
/// * `analysis` - Image analysis from `crate::analysis::analyze_image`
/// * `target_z` - Target unified quality (0-100, where Z ≈ SSIM2)
/// * `optimize_for` - Which metric to optimize for
///
/// # Example
///
/// ```ignore
/// use zenjpeg_dispatch::{analyze_image, OptimizeFor};
/// use zenjpeg_dispatch::unified_quality::select_codec_for_z;
///
/// let analysis = analyze_image(&pixels, width, height, 75.0);
/// let codec = select_codec_for_z(&analysis, 75.0, OptimizeFor::Butteraugli);
/// ```
#[must_use]
pub fn select_codec_for_z(
    analysis: &crate::analysis::ImageAnalysis,
    target_z: f32,
    optimize_for: crate::types::OptimizeFor,
) -> crate::analysis::CodecRecommendation {
    use crate::analysis::{
        select_codec_for_butteraugli, select_codec_for_dssim, CodecRecommendation,
    };
    use crate::types::{OptimizeFor, Subsampling};

    match optimize_for {
        OptimizeFor::Butteraugli => {
            let target_ba = unified_quality_to_target_butteraugli(target_z as u8);
            select_codec_for_butteraugli(analysis, target_ba)
        }
        OptimizeFor::Dssim => {
            let target_dssim = unified_quality_to_target_dssim(target_z as u8);
            select_codec_for_dssim(analysis, target_dssim)
        }
        OptimizeFor::Ssimulacra2 => {
            // SSIM2 correlates strongly with Butteraugli (r = -0.88)
            // Use Butteraugli heuristic as proxy
            let target_ba = unified_quality_to_target_butteraugli(target_z as u8);
            select_codec_for_butteraugli(analysis, target_ba)
        }
        OptimizeFor::FileSize => {
            // For pure file size optimization, mozjpeg-420 wins
            CodecRecommendation::MozJpeg {
                subsampling: Subsampling::S420,
            }
        }
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
        let luma: Vec<f32> = pixels
            .chunks(3)
            .map(|rgb| 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
            .collect();

        let mean: f32 = luma.iter().sum::<f32>() / luma.len() as f32;
        analysis.variance =
            luma.iter().map(|&l| (l - mean).powi(2)).sum::<f32>() / luma.len() as f32;

        // Estimate edge density using simple gradient
        let mut edge_sum = 0.0f32;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
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

        self.variance > 500.0 && self.chroma_complexity > 0.1 && self.uniform_block_fraction < 0.3
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
            assert!(
                bpp >= prev_bpp,
                "bpp should increase with quality: Q{} bpp={} < prev={}",
                q,
                bpp,
                prev_bpp
            );
            prev_bpp = bpp;
        }
    }

    #[test]
    fn test_unified_quality_to_butteraugli_monotonic() {
        let mut prev_ba = f32::MAX;
        for q in 0..=100 {
            let ba = unified_quality_to_target_butteraugli(q);
            assert!(
                ba <= prev_ba,
                "butteraugli should decrease with quality: Q{} ba={} > prev={}",
                q,
                ba,
                prev_ba
            );
            prev_ba = ba;
        }
    }

    #[test]
    fn test_unified_quality_to_ssimulacra2_monotonic() {
        let mut prev_ssim2 = 0.0;
        for q in 0..=100 {
            let ssim2 = unified_quality_to_target_ssimulacra2(q);
            assert!(
                ssim2 >= prev_ssim2,
                "ssim2 should increase with quality: Q{} ssim2={} < prev={}",
                q,
                ssim2,
                prev_ssim2
            );
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
                pixels[idx] = (x * 4) as u8; // R gradient
                pixels[idx + 1] = (y * 4) as u8; // G gradient
                pixels[idx + 2] = 128; // B constant
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

    #[test]
    fn test_dssim_mapping_monotonic() {
        let mut prev_dssim = f32::MAX;
        for q in 0..=100 {
            let dssim = unified_quality_to_target_dssim(q);
            assert!(
                dssim <= prev_dssim,
                "dssim should decrease with quality: Q{} dssim={} > prev={}",
                q,
                dssim,
                prev_dssim
            );
            prev_dssim = dssim;
        }
    }

    #[test]
    fn test_butteraugli_roundtrip() {
        // Test roundtrip conversion
        for z in [0, 25, 50, 75, 90, 100] {
            let ba = unified_quality_to_target_butteraugli(z);
            let z_back = butteraugli_to_unified_quality(ba);
            assert!(
                (z_back - z as f32).abs() < 1.0,
                "Roundtrip failed: z={}, ba={}, z_back={}",
                z,
                ba,
                z_back
            );
        }
    }

    #[test]
    fn test_dssim_roundtrip() {
        // Test roundtrip conversion
        for z in [0, 25, 50, 75, 90, 100] {
            let dssim = unified_quality_to_target_dssim(z);
            let z_back = dssim_to_unified_quality(dssim);
            assert!(
                (z_back - z as f32).abs() < 1.0,
                "Roundtrip failed: z={}, dssim={}, z_back={}",
                z,
                dssim,
                z_back
            );
        }
    }

    #[test]
    fn test_ssim2_identity() {
        // Z = SSIM2 by design
        for z in 0..=100 {
            let ssim2 = unified_quality_to_target_ssimulacra2(z);
            assert_eq!(ssim2, z as f32, "Z should equal SSIM2");
        }
    }

    #[test]
    fn test_select_codec_for_z() {
        use crate::analysis::analyze_image;
        use crate::types::OptimizeFor;

        let pixels = vec![128u8; 64 * 64 * 3];
        let analysis = analyze_image(&pixels, 64, 64, 75.0);

        // For Butteraugli at high Z, should recommend jpegli
        let rec = select_codec_for_z(&analysis, 80.0, OptimizeFor::Butteraugli);
        assert!(rec.is_jpegli());

        // For DSSIM at same Z, should recommend mozjpeg (opposite preference)
        let rec = select_codec_for_z(&analysis, 80.0, OptimizeFor::Dssim);
        assert!(rec.is_mozjpeg());

        // For FileSize, always mozjpeg-420
        let rec = select_codec_for_z(&analysis, 80.0, OptimizeFor::FileSize);
        assert!(rec.is_mozjpeg());
        assert_eq!(rec.subsampling(), crate::types::Subsampling::S420);
    }

    #[test]
    fn test_data_driven_values() {
        // Verify the data-driven mappings match expected values from analysis
        // At Z=50: BA ≈ 4.4, DSSIM ≈ 0.0062
        let ba_50 = unified_quality_to_target_butteraugli(50);
        assert!(
            ba_50 > 4.0 && ba_50 < 5.0,
            "BA at Z=50 should be ~4.4, got {}",
            ba_50
        );

        let dssim_50 = unified_quality_to_target_dssim(50);
        assert!(
            dssim_50 > 0.005 && dssim_50 < 0.008,
            "DSSIM at Z=50 should be ~0.0062, got {}",
            dssim_50
        );

        // At Z=75: BA ≈ 3.1, DSSIM ≈ 0.0033
        let ba_75 = unified_quality_to_target_butteraugli(75);
        assert!(
            ba_75 > 2.5 && ba_75 < 3.5,
            "BA at Z=75 should be ~3.1, got {}",
            ba_75
        );

        let dssim_75 = unified_quality_to_target_dssim(75);
        assert!(
            dssim_75 > 0.002 && dssim_75 < 0.005,
            "DSSIM at Z=75 should be ~0.0033, got {}",
            dssim_75
        );
    }
}
