//! Benchmark SSIMULACRA2 vs Butteraugli computation time

use codec_eval::metrics::butteraugli::calculate_butteraugli;
use codec_eval::metrics::ssimulacra2::calculate_ssimulacra2;
use std::time::Instant;

fn main() {
    // Test different image sizes
    let sizes = [(256, 256), (512, 512), (768, 512), (1024, 1024)];
    let iterations = 10;

    println!(
        "Metric Timing Benchmark ({} iterations per size)\n",
        iterations
    );
    println!(
        "{:>12} {:>15} {:>15} {:>10}",
        "Size", "Butteraugli", "SSIMULACRA2", "Ratio"
    );
    println!("{:-<55}", "");

    for (width, height) in sizes {
        // Create source and distorted images (RGB u8)
        let mut source = vec![0u8; width * height * 3];
        let mut distorted = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                source[idx] = (x % 256) as u8;
                source[idx + 1] = (y % 256) as u8;
                source[idx + 2] = ((x + y) % 256) as u8;
                // Add some distortion
                distorted[idx] = source[idx].saturating_add(5);
                distorted[idx + 1] = source[idx + 1].saturating_add(3);
                distorted[idx + 2] = source[idx + 2].saturating_add(4);
            }
        }

        // Benchmark Butteraugli
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = calculate_butteraugli(&source, &distorted, width, height);
        }
        let ba_time = start.elapsed() / iterations as u32;

        // Benchmark SSIMULACRA2
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = calculate_ssimulacra2(&source, &distorted, width, height);
        }
        let ssim2_time = start.elapsed() / iterations as u32;

        let ratio = ssim2_time.as_secs_f64() / ba_time.as_secs_f64();

        println!(
            "{:>5}x{:<5} {:>12.2?} {:>15.2?} {:>9.2}x",
            width, height, ba_time, ssim2_time, ratio
        );
    }
}
