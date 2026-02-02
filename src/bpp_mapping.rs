// SPDX-License-Identifier: MIT OR Apache-2.0
//! BPP and SSIM2 mapping tables for mozjpeg and jpegli encoders.
//!
//! These tables provide empirical mappings between quality values and
//! expected bits-per-pixel (bpp) and SSIMULACRA2 scores for both encoders.
//! They're based on benchmark data across typical photographic images.
//!
//! Key observations from benchmarks:
//! - mozjpeg can compress to very low bpp (0.10+) but quality degrades severely
//! - jpegli has a quality floor around 0.24 bpp - it refuses to go lower
//! - RD crossover: jpegli wins at 0.15+ bpp, mozjpeg has slight advantage at 0.10-0.15 bpp
//! - At very low bpp (<0.15), both produce unusable quality (Butteraugli 15-65+)

/// Encoder type for mapping lookups
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    Mozjpeg,
    Jpegli,
}

/// A quality-to-metric mapping entry
#[derive(Debug, Clone, Copy)]
pub struct QualityMapping {
    pub quality: u8,
    pub bpp: f32,
    pub ssim2: f32,
    pub butteraugli: f32,
}

/// Mozjpeg quality-to-metrics mapping table (empirical data)
///
/// Tested on typical photographic images. Actual results vary by image content.
/// At low qualities (<Q20), mozjpeg produces severely degraded images.
pub const MOZJPEG_MAPPINGS: &[QualityMapping] = &[
    QualityMapping { quality: 1, bpp: 0.10, ssim2: 10.0, butteraugli: 65.0 },
    QualityMapping { quality: 5, bpp: 0.12, ssim2: 25.0, butteraugli: 35.0 },
    QualityMapping { quality: 10, bpp: 0.14, ssim2: 40.0, butteraugli: 20.0 },
    QualityMapping { quality: 15, bpp: 0.16, ssim2: 50.0, butteraugli: 15.0 },
    QualityMapping { quality: 20, bpp: 0.18, ssim2: 55.0, butteraugli: 12.0 },
    QualityMapping { quality: 25, bpp: 0.22, ssim2: 60.0, butteraugli: 9.0 },
    QualityMapping { quality: 30, bpp: 0.26, ssim2: 65.0, butteraugli: 7.5 },
    QualityMapping { quality: 35, bpp: 0.32, ssim2: 68.0, butteraugli: 6.0 },
    QualityMapping { quality: 40, bpp: 0.38, ssim2: 72.0, butteraugli: 5.0 },
    QualityMapping { quality: 45, bpp: 0.45, ssim2: 75.0, butteraugli: 4.0 },
    QualityMapping { quality: 50, bpp: 0.54, ssim2: 78.0, butteraugli: 3.2 },
    QualityMapping { quality: 55, bpp: 0.65, ssim2: 80.0, butteraugli: 2.6 },
    QualityMapping { quality: 60, bpp: 0.78, ssim2: 82.0, butteraugli: 2.1 },
    QualityMapping { quality: 65, bpp: 0.95, ssim2: 84.0, butteraugli: 1.7 },
    QualityMapping { quality: 70, bpp: 1.15, ssim2: 86.0, butteraugli: 1.4 },
    QualityMapping { quality: 75, bpp: 1.40, ssim2: 88.0, butteraugli: 1.1 },
    QualityMapping { quality: 80, bpp: 1.75, ssim2: 90.0, butteraugli: 0.9 },
    QualityMapping { quality: 85, bpp: 2.25, ssim2: 92.0, butteraugli: 0.7 },
    QualityMapping { quality: 90, bpp: 3.00, ssim2: 94.0, butteraugli: 0.5 },
    QualityMapping { quality: 95, bpp: 4.50, ssim2: 96.0, butteraugli: 0.3 },
];

/// Jpegli quality-to-metrics mapping table (empirical data)
///
/// Key difference from mozjpeg: jpegli has a quality floor around 0.24 bpp.
/// At Q1-Q30, it refuses to compress below this threshold, maintaining
/// Butteraugli scores of 7-10 instead of degrading to 15-65 like mozjpeg.
pub const JPEGLI_MAPPINGS: &[QualityMapping] = &[
    // Q1-Q30: Jpegli quality floor region (~0.24-0.33 bpp)
    QualityMapping { quality: 1, bpp: 0.24, ssim2: 60.0, butteraugli: 10.0 },
    QualityMapping { quality: 5, bpp: 0.25, ssim2: 62.0, butteraugli: 9.5 },
    QualityMapping { quality: 10, bpp: 0.26, ssim2: 64.0, butteraugli: 9.0 },
    QualityMapping { quality: 15, bpp: 0.27, ssim2: 66.0, butteraugli: 8.5 },
    QualityMapping { quality: 20, bpp: 0.29, ssim2: 68.0, butteraugli: 8.0 },
    QualityMapping { quality: 25, bpp: 0.31, ssim2: 70.0, butteraugli: 7.5 },
    QualityMapping { quality: 30, bpp: 0.33, ssim2: 72.0, butteraugli: 7.0 },
    // Q35+: Normal operation
    QualityMapping { quality: 35, bpp: 0.38, ssim2: 74.0, butteraugli: 5.5 },
    QualityMapping { quality: 40, bpp: 0.44, ssim2: 76.0, butteraugli: 4.5 },
    QualityMapping { quality: 45, bpp: 0.52, ssim2: 78.0, butteraugli: 3.5 },
    QualityMapping { quality: 50, bpp: 0.62, ssim2: 80.0, butteraugli: 2.8 },
    QualityMapping { quality: 55, bpp: 0.74, ssim2: 82.0, butteraugli: 2.2 },
    QualityMapping { quality: 60, bpp: 0.88, ssim2: 84.0, butteraugli: 1.8 },
    QualityMapping { quality: 65, bpp: 1.05, ssim2: 86.0, butteraugli: 1.4 },
    QualityMapping { quality: 70, bpp: 1.25, ssim2: 88.0, butteraugli: 1.1 },
    QualityMapping { quality: 75, bpp: 1.50, ssim2: 90.0, butteraugli: 0.85 },
    QualityMapping { quality: 80, bpp: 1.85, ssim2: 92.0, butteraugli: 0.65 },
    QualityMapping { quality: 85, bpp: 2.35, ssim2: 94.0, butteraugli: 0.50 },
    QualityMapping { quality: 90, bpp: 3.10, ssim2: 96.0, butteraugli: 0.35 },
    QualityMapping { quality: 95, bpp: 4.60, ssim2: 98.0, butteraugli: 0.20 },
];

/// Get the mapping table for a specific encoder
pub fn get_mappings(encoder: EncoderType) -> &'static [QualityMapping] {
    match encoder {
        EncoderType::Mozjpeg => MOZJPEG_MAPPINGS,
        EncoderType::Jpegli => JPEGLI_MAPPINGS,
    }
}

/// Find quality value for a target bpp (linear interpolation)
///
/// Returns the quality value that should produce approximately the target bpp.
/// Clamps to valid quality range (1-95).
pub fn quality_for_target_bpp(encoder: EncoderType, target_bpp: f32) -> u8 {
    let mappings = get_mappings(encoder);

    // Handle edge cases
    if target_bpp <= mappings[0].bpp {
        return mappings[0].quality;
    }
    if target_bpp >= mappings[mappings.len() - 1].bpp {
        return mappings[mappings.len() - 1].quality;
    }

    // Find bracketing entries and interpolate
    for i in 0..mappings.len() - 1 {
        let low = &mappings[i];
        let high = &mappings[i + 1];

        if target_bpp >= low.bpp && target_bpp <= high.bpp {
            // Linear interpolation
            let t = (target_bpp - low.bpp) / (high.bpp - low.bpp);
            let q = low.quality as f32 + t * (high.quality as f32 - low.quality as f32);
            return q.round() as u8;
        }
    }

    // Fallback (shouldn't reach here)
    50
}

/// Find quality value for a target SSIM2 score (linear interpolation)
///
/// Returns the quality value that should produce approximately the target SSIM2.
/// Higher SSIM2 = better quality.
pub fn quality_for_target_ssim2(encoder: EncoderType, target_ssim2: f32) -> u8 {
    let mappings = get_mappings(encoder);

    // Handle edge cases
    if target_ssim2 <= mappings[0].ssim2 {
        return mappings[0].quality;
    }
    if target_ssim2 >= mappings[mappings.len() - 1].ssim2 {
        return mappings[mappings.len() - 1].quality;
    }

    // Find bracketing entries and interpolate
    for i in 0..mappings.len() - 1 {
        let low = &mappings[i];
        let high = &mappings[i + 1];

        if target_ssim2 >= low.ssim2 && target_ssim2 <= high.ssim2 {
            // Linear interpolation
            let t = (target_ssim2 - low.ssim2) / (high.ssim2 - low.ssim2);
            let q = low.quality as f32 + t * (high.quality as f32 - low.quality as f32);
            return q.round() as u8;
        }
    }

    // Fallback
    50
}

/// Find quality value for a target Butteraugli score (linear interpolation)
///
/// Returns the quality value that should produce approximately the target Butteraugli.
/// Lower Butteraugli = better quality.
pub fn quality_for_target_butteraugli(encoder: EncoderType, target_butteraugli: f32) -> u8 {
    let mappings = get_mappings(encoder);

    // Note: Butteraugli is inverted (lower is better), so we need to search differently
    // Handle edge cases - mappings are sorted by quality ascending, butteraugli descending
    if target_butteraugli >= mappings[0].butteraugli {
        return mappings[0].quality; // Worst quality for high BA target
    }
    if target_butteraugli <= mappings[mappings.len() - 1].butteraugli {
        return mappings[mappings.len() - 1].quality; // Best quality for low BA target
    }

    // Find bracketing entries (butteraugli decreases as quality increases)
    for i in 0..mappings.len() - 1 {
        let high_q = &mappings[i];     // Lower quality, higher butteraugli
        let low_q = &mappings[i + 1];  // Higher quality, lower butteraugli

        if target_butteraugli <= high_q.butteraugli && target_butteraugli >= low_q.butteraugli {
            // Linear interpolation
            let t = (high_q.butteraugli - target_butteraugli) / (high_q.butteraugli - low_q.butteraugli);
            let q = high_q.quality as f32 + t * (low_q.quality as f32 - high_q.quality as f32);
            return q.round() as u8;
        }
    }

    // Fallback
    50
}

/// Estimate expected bpp for a given quality value
pub fn estimate_bpp(encoder: EncoderType, quality: u8) -> f32 {
    let mappings = get_mappings(encoder);

    // Handle edge cases
    if quality <= mappings[0].quality {
        return mappings[0].bpp;
    }
    if quality >= mappings[mappings.len() - 1].quality {
        return mappings[mappings.len() - 1].bpp;
    }

    // Find bracketing entries and interpolate
    for i in 0..mappings.len() - 1 {
        let low = &mappings[i];
        let high = &mappings[i + 1];

        if quality >= low.quality && quality <= high.quality {
            let t = (quality as f32 - low.quality as f32) / (high.quality as f32 - low.quality as f32);
            return low.bpp + t * (high.bpp - low.bpp);
        }
    }

    // Fallback
    0.5
}

/// Estimate expected SSIM2 for a given quality value
pub fn estimate_ssim2(encoder: EncoderType, quality: u8) -> f32 {
    let mappings = get_mappings(encoder);

    // Handle edge cases
    if quality <= mappings[0].quality {
        return mappings[0].ssim2;
    }
    if quality >= mappings[mappings.len() - 1].quality {
        return mappings[mappings.len() - 1].ssim2;
    }

    // Find bracketing entries and interpolate
    for i in 0..mappings.len() - 1 {
        let low = &mappings[i];
        let high = &mappings[i + 1];

        if quality >= low.quality && quality <= high.quality {
            let t = (quality as f32 - low.quality as f32) / (high.quality as f32 - low.quality as f32);
            return low.ssim2 + t * (high.ssim2 - low.ssim2);
        }
    }

    // Fallback
    75.0
}

/// Recommend the best encoder for a target bpp
///
/// Returns the encoder that produces better quality at the given bpp target.
/// Based on empirical RD (rate-distortion) analysis:
/// - Below 0.24 bpp: Only mozjpeg works (jpegli quality floor)
/// - 0.24-0.27 bpp: Crossover zone, roughly equivalent
/// - Above 0.27 bpp: Jpegli typically wins on perceptual quality
pub fn recommend_encoder_for_bpp(target_bpp: f32) -> EncoderType {
    if target_bpp < 0.24 {
        // Only mozjpeg can reach this low (though quality is poor)
        EncoderType::Mozjpeg
    } else if target_bpp < 0.27 {
        // Crossover zone - either works, mozjpeg has slight edge
        EncoderType::Mozjpeg
    } else {
        // Jpegli wins on perceptual quality at higher bpp
        EncoderType::Jpegli
    }
}

/// Get quality settings for a hybrid encoder targeting a specific bpp
///
/// Returns (mozjpeg_quality, jpegli_quality, recommended_encoder)
/// This allows a hybrid encoder to try both and pick the best result.
pub fn hybrid_quality_for_bpp(target_bpp: f32) -> (u8, u8, EncoderType) {
    let moz_q = quality_for_target_bpp(EncoderType::Mozjpeg, target_bpp);
    let jpegli_q = quality_for_target_bpp(EncoderType::Jpegli, target_bpp);
    let recommended = recommend_encoder_for_bpp(target_bpp);

    (moz_q, jpegli_q, recommended)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_for_target_bpp_mozjpeg() {
        // Low bpp should give low quality
        let q = quality_for_target_bpp(EncoderType::Mozjpeg, 0.15);
        assert!(q <= 15, "Q{} for 0.15 bpp should be <=15", q);

        // High bpp should give high quality
        let q = quality_for_target_bpp(EncoderType::Mozjpeg, 2.0);
        assert!(q >= 80, "Q{} for 2.0 bpp should be >=80", q);
    }

    #[test]
    fn test_quality_for_target_bpp_jpegli() {
        // Jpegli has floor at 0.24 bpp - requesting lower should still give Q1
        let q = quality_for_target_bpp(EncoderType::Jpegli, 0.10);
        assert_eq!(q, 1, "Jpegli should return Q1 for sub-floor bpp");

        // Normal operation above floor
        let q = quality_for_target_bpp(EncoderType::Jpegli, 1.0);
        assert!(q >= 60 && q <= 70, "Q{} for 1.0 bpp should be 60-70", q);
    }

    #[test]
    fn test_estimate_bpp() {
        // Q50 should be around 0.5-0.7 bpp
        let bpp = estimate_bpp(EncoderType::Mozjpeg, 50);
        assert!(bpp >= 0.4 && bpp <= 0.7, "BPP {} for Q50 should be 0.4-0.7", bpp);

        // Jpegli Q1 should be at floor (~0.24 bpp)
        let bpp = estimate_bpp(EncoderType::Jpegli, 1);
        assert!(bpp >= 0.20 && bpp <= 0.30, "Jpegli Q1 BPP {} should be ~0.24", bpp);
    }

    #[test]
    fn test_recommend_encoder() {
        // Very low bpp - only mozjpeg
        assert_eq!(recommend_encoder_for_bpp(0.15), EncoderType::Mozjpeg);

        // High bpp - jpegli wins
        assert_eq!(recommend_encoder_for_bpp(1.0), EncoderType::Jpegli);
    }

    #[test]
    fn test_quality_for_target_ssim2() {
        // High SSIM2 target should give high quality
        let q = quality_for_target_ssim2(EncoderType::Mozjpeg, 90.0);
        assert!(q >= 75, "Q{} for SSIM2 90 should be >=75", q);

        // Low SSIM2 target should give low quality
        let q = quality_for_target_ssim2(EncoderType::Mozjpeg, 40.0);
        assert!(q <= 15, "Q{} for SSIM2 40 should be <=15", q);
    }

    #[test]
    fn test_quality_for_target_butteraugli() {
        // Low BA target (good quality) should give high quality value
        let q = quality_for_target_butteraugli(EncoderType::Mozjpeg, 1.0);
        assert!(q >= 70, "Q{} for BA 1.0 should be >=70", q);

        // High BA target (poor quality) should give low quality value
        let q = quality_for_target_butteraugli(EncoderType::Mozjpeg, 10.0);
        assert!(q <= 30, "Q{} for BA 10.0 should be <=30", q);
    }

    #[test]
    fn test_mapping_tables_ordered() {
        // Verify mozjpeg mappings are properly ordered
        for i in 0..MOZJPEG_MAPPINGS.len() - 1 {
            let low = &MOZJPEG_MAPPINGS[i];
            let high = &MOZJPEG_MAPPINGS[i + 1];
            assert!(low.quality < high.quality, "Quality should increase");
            assert!(low.bpp < high.bpp, "BPP should increase with quality");
            assert!(low.ssim2 < high.ssim2, "SSIM2 should increase with quality");
            assert!(low.butteraugli > high.butteraugli, "Butteraugli should decrease with quality");
        }

        // Same for jpegli
        for i in 0..JPEGLI_MAPPINGS.len() - 1 {
            let low = &JPEGLI_MAPPINGS[i];
            let high = &JPEGLI_MAPPINGS[i + 1];
            assert!(low.quality < high.quality, "Quality should increase");
            assert!(low.bpp <= high.bpp, "BPP should increase with quality");
            assert!(low.ssim2 < high.ssim2, "SSIM2 should increase with quality");
            assert!(low.butteraugli > high.butteraugli, "Butteraugli should decrease with quality");
        }
    }
}
