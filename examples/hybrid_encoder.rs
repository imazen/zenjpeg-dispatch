//! Hybrid encoder prototype: jpegli AQ + mozjpeg trellis
//!
//! This combines:
//! - jpegli's adaptive quantization (global image analysis, per-block strength)
//! - mozjpeg's trellis quantization (per-coefficient rate-distortion optimization)
//!
//! Theory: AQ decides WHERE to spend bits, trellis decides HOW to spend them optimally.

use std::time::Instant;

// Access jpegli's adaptive quantization (marked #[doc(hidden)] but pub)
use jpegli::adaptive_quant::compute_aq_strength_map;

/// Convert RGB to Y plane (luminance only, for AQ analysis)
fn rgb_to_y_plane(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    let mut y_plane = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        // BT.601 luma
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        y_plane.push(y);
    }
    y_plane
}

/// Scale quantization table by AQ strength
fn scale_quant_table(base: &[u16; 64], aq_strength: f32) -> [u16; 64] {
    let mut scaled = [0u16; 64];
    // Higher AQ strength = more compression = higher quant values
    // aq_strength typically ranges from 0.0 to ~0.5
    // We map this to a multiplier: strength=0 → 1.0x, strength=0.5 → ~1.5x
    let multiplier = 1.0 + aq_strength;
    for i in 0..64 {
        scaled[i] = ((base[i] as f32 * multiplier).round() as u16)
            .max(1)
            .min(255);
    }
    scaled
}

/// Standard luminance quantization table
const STD_LUMA_QUANT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Scale quant table for quality level (simplified)
fn scale_quant_for_quality(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let scale = if quality < 50 {
        5000.0 / quality as f32
    } else {
        200.0 - 2.0 * quality as f32
    } / 100.0;

    let mut scaled = [0u16; 64];
    for i in 0..64 {
        let val = (base[i] as f32 * scale).round() as u16;
        scaled[i] = val.max(1).min(255);
    }
    scaled
}

fn main() {
    // Create test image (512x512 with patterns)
    let width = 512;
    let height = 512;
    let mut pixels = vec![0u8; width * height * 3];

    // Create image with varying content: smooth gradients + sharp edges
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;

            // Quadrant-based pattern
            let qx = x / (width / 2);
            let qy = y / (height / 2);

            match (qx, qy) {
                (0, 0) => {
                    // Smooth gradient
                    let val = ((x + y) * 255 / (width + height)) as u8;
                    pixels[idx] = val;
                    pixels[idx + 1] = val;
                    pixels[idx + 2] = val;
                }
                (1, 0) => {
                    // Sharp edges (checkerboard)
                    let val = if (x / 16 + y / 16) % 2 == 0 { 240 } else { 16 };
                    pixels[idx] = val;
                    pixels[idx + 1] = val;
                    pixels[idx + 2] = val;
                }
                (0, 1) => {
                    // Color gradient
                    pixels[idx] = (x % 256) as u8;
                    pixels[idx + 1] = (y % 256) as u8;
                    pixels[idx + 2] = 128;
                }
                (1, 1) => {
                    // High frequency noise-like pattern
                    let val = ((x * 7 + y * 13) % 256) as u8;
                    pixels[idx] = val;
                    pixels[idx + 1] = val;
                    pixels[idx + 2] = val;
                }
                _ => {}
            }
        }
    }

    let quality = 75u8;
    let base_quant = scale_quant_for_quality(&STD_LUMA_QUANT, quality);

    println!("=== Hybrid Encoder Prototype ===\n");
    println!("Image: {}x{}, Quality: {}", width, height, quality);

    // Step 1: Compute Y plane for AQ analysis
    let start = Instant::now();
    let y_plane = rgb_to_y_plane(&pixels, width, height);
    println!("\n1. RGB to Y plane: {:?}", start.elapsed());

    // Step 2: Compute AQ strength map using jpegli's algorithm
    let start = Instant::now();
    let y_quant_01 = base_quant[1]; // First AC coefficient quant value
    let aq_map = compute_aq_strength_map(&y_plane, width, height, y_quant_01);
    println!("2. Compute AQ map: {:?}", start.elapsed());

    // Analyze AQ map
    let width_blocks = (width + 7) / 8;
    let height_blocks = (height + 7) / 8;
    let mut aq_min = f32::MAX;
    let mut aq_max = f32::MIN;
    let mut aq_sum = 0.0f32;
    for by in 0..height_blocks {
        for bx in 0..width_blocks {
            let strength = aq_map.get(bx, by);
            aq_min = aq_min.min(strength);
            aq_max = aq_max.max(strength);
            aq_sum += strength;
        }
    }
    let aq_mean = aq_sum / (width_blocks * height_blocks) as f32;
    println!(
        "   AQ strengths: min={:.3}, max={:.3}, mean={:.3}",
        aq_min, aq_max, aq_mean
    );

    // Step 3: Show AQ map for different image regions
    println!("\n3. AQ strength by quadrant:");
    let quadrants = [
        ("Smooth gradient (top-left)", 0, 0),
        ("Sharp edges (top-right)", width_blocks / 2, 0),
        ("Color gradient (bottom-left)", 0, height_blocks / 2),
        (
            "High frequency (bottom-right)",
            width_blocks / 2,
            height_blocks / 2,
        ),
    ];

    for (name, start_bx, start_by) in quadrants {
        let mut sum = 0.0f32;
        let mut count = 0;
        for by in start_by..(start_by + 8).min(height_blocks) {
            for bx in start_bx..(start_bx + 8).min(width_blocks) {
                sum += aq_map.get(bx, by);
                count += 1;
            }
        }
        let mean = sum / count as f32;
        println!("   {}: mean AQ = {:.3}", name, mean);
    }

    // Step 4: Demonstrate per-block quant table scaling
    println!("\n4. Quant table scaling examples:");
    let low_aq = aq_map.get(2, 2); // Likely smooth area
    let high_aq = aq_map.get(width_blocks / 2 + 2, 2); // Likely edge area

    let scaled_low = scale_quant_table(&base_quant, low_aq);
    let scaled_high = scale_quant_table(&base_quant, high_aq);

    println!("   Base quant[0..4]: {:?}", &base_quant[0..4]);
    println!("   Low AQ ({:.3}) scaled: {:?}", low_aq, &scaled_low[0..4]);
    println!(
        "   High AQ ({:.3}) scaled: {:?}",
        high_aq,
        &scaled_high[0..4]
    );

    // Step 5: Compare approaches
    println!("\n5. Encoding comparison:");

    // Encode with standard zenjpeg (mozjpeg strategy)
    let start = Instant::now();
    let mozjpeg_result = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(quality))
        .strategy(zenjpeg::EncodingStrategy::Mozjpeg)
        .encode_rgb(&pixels, width, height)
        .unwrap();
    let mozjpeg_time = start.elapsed();

    // Encode with jpegli
    let start = Instant::now();
    let jpegli_result = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(quality))
        .strategy(zenjpeg::EncodingStrategy::Jpegli)
        .encode_rgb(&pixels, width, height)
        .unwrap();
    let jpegli_time = start.elapsed();

    println!(
        "   mozjpeg: {} bytes in {:?}",
        mozjpeg_result.len(),
        mozjpeg_time
    );
    println!(
        "   jpegli:  {} bytes in {:?}",
        jpegli_result.len(),
        jpegli_time
    );

    // Note about hybrid implementation
    println!("\n=== Hybrid Implementation Notes ===");
    println!(
        "
To fully implement hybrid encoding, we need to:

1. Extract Y/Cb/Cr planes from RGB
2. Compute AQ map from Y plane (done above)
3. For each 8x8 block:
   a. Get AQ strength for this block
   b. Scale quant table: effective_quant = base_quant * (1 + aq_strength)
   c. Run forward DCT
   d. Run trellis quantization with scaled quant table
   e. Store quantized coefficients
4. Run Huffman optimization on all coefficients
5. Encode to JPEG bitstream

The key insight is that trellis sees DIFFERENT quant tables per block,
making it spend fewer bits on busy/textured areas (high AQ) and more
bits on smooth/important areas (low AQ).
"
    );

    // Quick simulation: how much would quant tables vary?
    println!("=== Quant Table Variation Analysis ===\n");
    let mut quant_dc_min = u16::MAX;
    let mut quant_dc_max = 0u16;
    let mut total_blocks = 0;

    for by in 0..height_blocks {
        for bx in 0..width_blocks {
            let aq = aq_map.get(bx, by);
            let scaled = scale_quant_table(&base_quant, aq);
            quant_dc_min = quant_dc_min.min(scaled[0]);
            quant_dc_max = quant_dc_max.max(scaled[0]);
            total_blocks += 1;
        }
    }

    println!(
        "DC quant value range: {} to {} (base: {})",
        quant_dc_min, quant_dc_max, base_quant[0]
    );
    println!("Total blocks: {}", total_blocks);
    println!(
        "\nThis {}% variation in quant tables could improve rate-distortion\n\
              by letting trellis make block-appropriate decisions.",
        ((quant_dc_max - quant_dc_min) as f32 / base_quant[0] as f32 * 100.0) as i32
    );
}
