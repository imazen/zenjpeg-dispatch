//! Harvard Simulated Annealing quantization tables
//!
//! These tables are from research by Hopkins, Mitzenmacher, and Wagner-Carena at Harvard.
//! They were trained using simulated annealing to optimize JPEG compression.
//!
//! Source: https://www.eecs.harvard.edu/~michaelm/SimAnneal/
//!
//! License: Public Domain
//!
//! The tables achieve 37-52% compression gains over standard JPEG tables
//! while maintaining quality (error ratios typically > 0.85).

/// Simulated annealing optimized tables for different quality levels.
/// These are "top" tables - best overall balance of quality and compression.
/// Luminance (Y channel) only - use standard tables for chrominance.

/// Q95 - High quality SA table
/// From top_table_quality_95.txt
#[rustfmt::skip]
pub const SA_LUMA_Q95: [u16; 64] = [
     8,  13,  16,  25,  39,  68,  95,  74,
    13,  19,  47,  17,  49,  65,  79,  69,
    19,  15,  36,  23,  67,  95,  79,  58,
    15,  39,  30,  85,  79, 127, 128,  77,
    27,  47,  55,  75, 122, 174, 117,  76,
    45,  87,  75,  86, 102, 167, 178, 105,
    71,  96, 113, 115, 139, 156, 151, 122,
   137, 131, 161, 140, 176, 115, 132, 125,
];

/// Q75 - Medium-high quality SA table (best overall according to research)
/// From top_table_quality_75.txt
#[rustfmt::skip]
pub const SA_LUMA_Q75: [u16; 64] = [
     8,  22,  39,  43,  74,  67,  63,  65,
    34,  55,  48,  51,  76,  65,  77,  55,
    33,  42,  50,  57,  77,  75,  68,  47,
    58,  44,  60,  82,  78, 123,  76,  68,
    50,  74,  71,  91,  84, 103,  91,  66,
    50,  52,  53,  71,  73,  72, 137,  97,
    48,  66,  82,  66, 102, 154, 119,  96,
   100,  72,  92,  86,  94,  82, 124,  90,
];

/// Q50 - Medium quality SA table
/// From top_table_quality_50.txt
#[rustfmt::skip]
pub const SA_LUMA_Q50: [u16; 64] = [
     8,  30,  64,  86, 105,  97,  95,  70,
    31,  58,  78,  99,  88,  79,  73,  65,
    52,  66,  93, 103,  81,  83,  86,  70,
    70,  87,  94,  79,  80,  79, 104,  71,
    75,  98,  71,  91,  74, 111, 111,  83,
    92,  81,  71,  71,  69, 101, 142,  93,
    84,  82,  79,  64,  83, 120, 113, 103,
   100,  93, 127, 115, 128, 116,  69, 108,
];

/// Q35 - Low quality SA table
/// From top_table_quality_35.txt
#[rustfmt::skip]
pub const SA_LUMA_Q35: [u16; 64] = [
    11,  35,  35,  53,  61,  63,  47,  68,
    35,  41,  39,  54,  54,  57,  44,  47,
    44,  51,  61,  65,  48,  62,  79,  63,
    38,  53,  60,  57,  58,  96, 102,  76,
    44,  58,  68,  57,  81, 100,  96,  76,
    74,  60,  62,  72,  98, 121,  94, 110,
    55,  47,  91,  89,  81, 130, 102, 105,
    96, 106, 107, 100,  79,  87,  97, 103,
];

/// "Best compression" variants - prioritize smaller files
/// These have slightly worse quality but better compression.

/// Q75 best compression variant
/// From best_compression_quality_75.txt
#[rustfmt::skip]
pub const SA_LUMA_Q75_COMPRESS: [u16; 64] = [
     8,  20,  49,  64,  98,  82,  75,  78,
    35,  45,  66,  90,  93,  88,  71,  62,
    46,  66,  91,  93,  80,  70,  68,  64,
    62,  80,  86,  94,  77,  83,  81,  55,
    61,  75,  86,  84,  93, 110, 140,  93,
    82,  82,  77,  74,  92, 123,  70,  87,
    94,  80,  98,  75,  83, 176, 140,  93,
    90, 118, 101,  63, 108, 115, 136, 101,
];

/// Q50 best compression variant
/// From best_compression_quality_50.txt
#[rustfmt::skip]
pub const SA_LUMA_Q50_COMPRESS: [u16; 64] = [
     8,  24,  43,  56,  70,  68,  73,  56,
    36,  38,  61,  54,  73,  68,  60,  60,
    34,  38,  63,  78,  73,  67,  90,  69,
    39,  56,  70,  78,  74,  96,  78,  78,
    54,  70,  84,  72,  74,  97, 119,  74,
    75,  69,  68,  65, 126, 108,  95,  77,
    62,  73,  76,  97, 121, 111, 105,  70,
    85,  91,  77,  81,  76,  79,  99,  77,
];

/// Q35 best compression variant
/// From best_compression_quality_35.txt
#[rustfmt::skip]
pub const SA_LUMA_Q35_COMPRESS: [u16; 64] = [
    13,  43,  69,  84,  85,  77,  79,  72,
    46,  63,  77,  72,  97,  80,  41,  52,
    58,  81,  76,  90,  93,  87,  60,  47,
    65,  73,  66,  78,  98,  87,  98,  87,
    64,  66,  81,  69, 110, 153, 131,  73,
    61,  58,  61,  81,  86,  99, 113,  96,
    54,  48, 118,  51,  77, 162, 107,  63,
    66,  73,  76, 112, 162, 135, 115,  95,
];

/// Select the NEAREST SA table based on quality level.
/// No interpolation - just picks the closest trained table.
///
/// The SA tables are already at their target quality, so they should NOT
/// be scaled by the standard quality formula. Instead, use them directly.
pub fn select_sa_table(quality: u8) -> &'static [u16; 64] {
    // Map quality to nearest trained table (35, 50, 75, 95)
    // Midpoints: 42.5, 62.5, 85
    match quality {
        0..=42 => &SA_LUMA_Q35,
        43..=62 => &SA_LUMA_Q50,
        63..=84 => &SA_LUMA_Q75,
        85.. => &SA_LUMA_Q95,
    }
}

/// Select SA table with exact matching only.
/// Returns None if quality doesn't match a trained level exactly.
pub fn select_sa_table_exact(quality: u8) -> Option<&'static [u16; 64]> {
    match quality {
        35 => Some(&SA_LUMA_Q35),
        50 => Some(&SA_LUMA_Q50),
        75 => Some(&SA_LUMA_Q75),
        95 => Some(&SA_LUMA_Q95),
        _ => None,
    }
}

/// Select best compression SA table based on quality level.
pub fn select_sa_table_compress(quality: u8) -> &'static [u16; 64] {
    match quality {
        0..=42 => &SA_LUMA_Q35,       // No compression variant at Q35
        43..=62 => &SA_LUMA_Q50_COMPRESS,
        63..=84 => &SA_LUMA_Q75_COMPRESS,
        85.. => &SA_LUMA_Q95,         // No compression variant at Q95
    }
}

/// Scale an SA table by a factor.
/// SA tables need interpolation between trained quality levels.
///
/// For example, Q60 would interpolate between Q50 and Q75 tables.
pub fn scale_sa_table(base: &[u16; 64], scale_factor: f32) -> [u16; 64] {
    let mut result = [0u16; 64];
    for i in 0..64 {
        let scaled = (base[i] as f32 * scale_factor).round();
        result[i] = (scaled as u16).clamp(1, 255);
    }
    result
}

/// Interpolate between two SA tables.
/// t=0.0 returns a, t=1.0 returns b.
pub fn interpolate_sa_tables(a: &[u16; 64], b: &[u16; 64], t: f32) -> [u16; 64] {
    let mut result = [0u16; 64];
    let t = t.clamp(0.0, 1.0);
    for i in 0..64 {
        let va = a[i] as f32;
        let vb = b[i] as f32;
        let v = va * (1.0 - t) + vb * t;
        result[i] = (v.round() as u16).clamp(1, 255);
    }
    result
}

/// Get SA table with interpolation for any quality level 1-100.
/// This provides smooth transitions between the trained quality levels.
pub fn get_interpolated_sa_table(quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as f32;

    // Define the trained quality points
    const Q35: f32 = 35.0;
    const Q50: f32 = 50.0;
    const Q75: f32 = 75.0;
    const Q95: f32 = 95.0;

    if q <= Q35 {
        // Below Q35: scale down the Q35 table
        // Standard JPEG scaling: scale = 5000/q for q < 50
        // At Q35: scale = 5000/35 ≈ 143
        // At Q1: scale = 5000/1 = 5000
        // Ratio: (5000/q) / (5000/35) = 35/q
        let scale = Q35 / q;
        scale_sa_table(&SA_LUMA_Q35, scale)
    } else if q <= Q50 {
        // Interpolate Q35 -> Q50
        let t = (q - Q35) / (Q50 - Q35);
        interpolate_sa_tables(&SA_LUMA_Q35, &SA_LUMA_Q50, t)
    } else if q <= Q75 {
        // Interpolate Q50 -> Q75
        let t = (q - Q50) / (Q75 - Q50);
        interpolate_sa_tables(&SA_LUMA_Q50, &SA_LUMA_Q75, t)
    } else if q <= Q95 {
        // Interpolate Q75 -> Q95
        let t = (q - Q75) / (Q95 - Q75);
        interpolate_sa_tables(&SA_LUMA_Q75, &SA_LUMA_Q95, t)
    } else {
        // Above Q95: scale up the Q95 table (reduce quantization)
        // At Q95: scale = 200 - 2*95 = 10
        // At Q100: scale = 200 - 2*100 = 0 (would be all 1s)
        // Ratio: (200-2*q) / (200-2*95) = (200-2*q) / 10
        let base_scale = 200.0 - 2.0 * Q95;  // 10
        let new_scale = 200.0 - 2.0 * q;      // 0 at Q100
        let scale = new_scale / base_scale;   // 0 at Q100
        scale_sa_table(&SA_LUMA_Q95, scale.max(0.01))  // Don't go to 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sa_table_selection() {
        assert_eq!(select_sa_table(30), &SA_LUMA_Q35);
        assert_eq!(select_sa_table(50), &SA_LUMA_Q50);
        assert_eq!(select_sa_table(75), &SA_LUMA_Q75);
        assert_eq!(select_sa_table(95), &SA_LUMA_Q95);
    }

    #[test]
    fn test_interpolation() {
        // At exact quality points, should return that table
        let q50 = get_interpolated_sa_table(50);
        assert_eq!(q50, SA_LUMA_Q50);

        let q75 = get_interpolated_sa_table(75);
        assert_eq!(q75, SA_LUMA_Q75);

        // Midpoint should be average
        let q62 = get_interpolated_sa_table(62);
        // Check that values are between Q50 and Q75
        for i in 0..64 {
            let min = SA_LUMA_Q50[i].min(SA_LUMA_Q75[i]);
            let max = SA_LUMA_Q50[i].max(SA_LUMA_Q75[i]);
            assert!(q62[i] >= min && q62[i] <= max,
                "q62[{}] = {}, expected between {} and {}", i, q62[i], min, max);
        }
    }

    #[test]
    fn test_scaling() {
        // Scaling by 1.0 should be identity
        let scaled = scale_sa_table(&SA_LUMA_Q75, 1.0);
        assert_eq!(scaled, SA_LUMA_Q75);

        // Scaling by 2.0 should double (clamped to 255)
        let scaled = scale_sa_table(&SA_LUMA_Q75, 2.0);
        for i in 0..64 {
            let expected = ((SA_LUMA_Q75[i] as f32 * 2.0).round() as u16).min(255);
            assert_eq!(scaled[i], expected);
        }
    }
}
