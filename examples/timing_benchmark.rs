//! Quick timing benchmark for encoding strategies

use std::time::Instant;
use zenjpeg::{Encoder, EncodingStrategy, Quality};

fn main() {
    // Create test images of different sizes
    let sizes = [(512, 512), (1024, 1024), (2048, 2048)];

    for (width, height) in sizes {
        let mut pixels = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                pixels[idx] = (x % 256) as u8;
                pixels[idx + 1] = (y % 256) as u8;
                pixels[idx + 2] = ((x + y) % 256) as u8;
            }
        }

        let iterations = if width >= 2048 { 5 } else { 10 };

        println!("\n{}x{} image, Q85, {} iterations:", width, height, iterations);

        // Mozjpeg strategy
        let start = Instant::now();
        let mut size = 0;
        for _ in 0..iterations {
            let encoder = Encoder::new()
                .quality(Quality::Standard(85))
                .strategy(EncodingStrategy::Mozjpeg);
            size = encoder.encode_rgb(&pixels, width, height).unwrap().len();
        }
        let mozjpeg_time = start.elapsed() / iterations as u32;
        println!("  Mozjpeg:  {:>8?}  ({} bytes)", mozjpeg_time, size);

        // Jpegli strategy
        let start = Instant::now();
        for _ in 0..iterations {
            let encoder = Encoder::new()
                .quality(Quality::Standard(85))
                .strategy(EncodingStrategy::Jpegli);
            size = encoder.encode_rgb(&pixels, width, height).unwrap().len();
        }
        let jpegli_time = start.elapsed() / iterations as u32;
        println!("  Jpegli:   {:>8?}  ({} bytes)", jpegli_time, size);

        // Max compression (mozjpeg progressive + trellis)
        let start = Instant::now();
        for _ in 0..iterations {
            let encoder = Encoder::max_compression();
            size = encoder.encode_rgb(&pixels, width, height).unwrap().len();
        }
        let max_time = start.elapsed() / iterations as u32;
        println!("  MaxCompr: {:>8?}  ({} bytes)", max_time, size);

        // Fastest
        let start = Instant::now();
        for _ in 0..iterations {
            let encoder = Encoder::fastest();
            size = encoder.encode_rgb(&pixels, width, height).unwrap().len();
        }
        let fast_time = start.elapsed() / iterations as u32;
        println!("  Fastest:  {:>8?}  ({} bytes)", fast_time, size);
    }
}
