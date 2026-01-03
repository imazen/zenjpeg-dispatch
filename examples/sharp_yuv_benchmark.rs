//! Benchmark comparing mozjpeg with standard vs sharp YUV chroma subsampling.
//!
//! Sharp YUV uses bi-linear interpolation and gamma correction for higher quality
//! chroma downsampling, which can reduce color fringing artifacts.
//!
//! Usage:
//!   cargo run --release --example sharp_yuv_benchmark
//!
//! Outputs:
//!   - comparison_outputs/sharp_yuv_results.csv

use codec_eval::{
    decode::jpeg_decode_callback, EvalConfig, EvalSession, ImageData, MetricConfig, RDPoint,
    ViewingCondition,
};
use mozjpeg_oxide::{Encoder, Preset, Subsampling};
use std::fs;
use std::path::PathBuf;
use yuv::{
    rgb_to_sharp_yuv420, rgb_to_yuv420, SharpYuvGammaTransfer, YuvChromaSubsampling,
    YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};

/// Encode RGB to JPEG using mozjpeg's internal color conversion
fn encode_mozjpeg_standard(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    Encoder::new(Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(Subsampling::S420)
        .encode_rgb(rgb, width, height)
        .expect("mozjpeg encode failed")
}

/// Encode RGB to JPEG using yuv crate's standard YUV420 conversion + mozjpeg planar API
fn encode_yuv_standard(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    // Allocate YUV planes using yuv crate's alloc method
    let mut planar = YuvPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);

    // Standard YUV conversion (BT.601 full range for JPEG)
    // Using Professional mode for highest precision
    rgb_to_yuv420(
        &mut planar,
        rgb,
        width * 3, // RGB stride
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        YuvConversionMode::Professional,
    )
    .expect("yuv conversion failed");

    // Extract plane data
    let y_plane: Vec<u8> = planar.y_plane.borrow().to_vec();
    let u_plane: Vec<u8> = planar.u_plane.borrow().to_vec();
    let v_plane: Vec<u8> = planar.v_plane.borrow().to_vec();

    // Encode using mozjpeg's planar API
    Encoder::new(Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(Subsampling::S420)
        .encode_ycbcr_planar(&y_plane, &u_plane, &v_plane, width, height)
        .expect("mozjpeg planar encode failed")
}

/// Encode RGB to JPEG using yuv crate's sharp YUV420 conversion + mozjpeg planar API
fn encode_sharp_yuv(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    // Allocate YUV planes using yuv crate's alloc method
    let mut planar = YuvPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);

    // Sharp YUV conversion with sRGB gamma correction (BT.601 full range for JPEG)
    rgb_to_sharp_yuv420(
        &mut planar,
        rgb,
        width * 3, // RGB stride
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        SharpYuvGammaTransfer::Srgb,
    )
    .expect("sharp yuv conversion failed");

    // Extract plane data
    let y_plane: Vec<u8> = planar.y_plane.borrow().to_vec();
    let u_plane: Vec<u8> = planar.u_plane.borrow().to_vec();
    let v_plane: Vec<u8> = planar.v_plane.borrow().to_vec();

    // Encode using mozjpeg's planar API
    Encoder::new(Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(Subsampling::S420)
        .encode_ycbcr_planar(&y_plane, &u_plane, &v_plane, width, height)
        .expect("mozjpeg planar encode failed")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = PathBuf::from("comparison_outputs");
    fs::create_dir_all(&output_dir)?;

    // Configure evaluation with perceptual metrics
    let config = EvalConfig::builder()
        .report_dir(&output_dir)
        .viewing(ViewingCondition::desktop())
        .metrics(MetricConfig::perceptual())
        .quality_levels(vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0,
        ])
        .build();

    let mut session = EvalSession::new(config);

    // Register mozjpeg with standard internal color conversion
    session.add_codec_with_decode(
        "mozjpeg-standard",
        "internal",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;

            Ok(encode_mozjpeg_standard(&rgb, width, height, quality))
        }),
        jpeg_decode_callback(),
    );

    // Register mozjpeg with yuv crate standard conversion (professional mode)
    session.add_codec_with_decode(
        "mozjpeg-yuv-pro",
        "yuv-0.8-pro",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;

            Ok(encode_yuv_standard(&rgb, width, height, quality))
        }),
        jpeg_decode_callback(),
    );

    // Register mozjpeg with sharp YUV conversion
    session.add_codec_with_decode(
        "mozjpeg-sharp-yuv",
        "yuv-0.8-sharp",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;

            Ok(encode_sharp_yuv(&rgb, width, height, quality))
        }),
        jpeg_decode_callback(),
    );

    println!("Registered {} codecs", session.codec_count());
    println!("  - mozjpeg-standard: internal color conversion");
    println!("  - mozjpeg-yuv-pro: yuv crate professional mode 4:2:0");
    println!("  - mozjpeg-sharp-yuv: yuv crate sharp 4:2:0 (bi-linear + sRGB gamma)");

    // Find test images
    let corpus_dir = PathBuf::from("../codec-eval/codec-corpus/kodak");
    let test_images: Vec<PathBuf> = if corpus_dir.exists() {
        fs::read_dir(&corpus_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "png"))
            .take(6)
            .collect()
    } else {
        println!(
            "Kodak corpus not found at {:?}, using synthetic image",
            corpus_dir
        );
        vec![]
    };

    let mut all_points: Vec<RDPoint> = Vec::new();

    if test_images.is_empty() {
        // Create synthetic test image with color gradients (good for chroma testing)
        println!("\nCreating synthetic color gradient image...");
        let width = 512usize;
        let height = 512usize;
        let mut rgb = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                // Diagonal color gradients to stress chroma subsampling
                rgb[idx] = ((x * 255) / width) as u8; // R
                rgb[idx + 1] = ((y * 255) / height) as u8; // G
                rgb[idx + 2] = (((x + y) * 128) / (width + height)) as u8; // B
            }
        }

        let image = ImageData::RgbSlice {
            data: rgb,
            width,
            height,
        };

        let report = session.evaluate_image("synthetic_gradient", image)?;
        println!(
            "\n=== {} ({}x{}) ===",
            report.name, report.width, report.height
        );

        for result in &report.results {
            let dssim = result.metrics.dssim.unwrap_or(0.0);
            let ssim2 = result.metrics.ssimulacra2.unwrap_or(0.0);
            let bfly = result.metrics.butteraugli.unwrap_or(0.0);
            println!(
                "{:20} Q{:>3}: {:>7} bytes, {:>5.2} bpp, SSIM2={:>5.1}, BA={:>5.2}, DSSIM={:.6}",
                result.codec_id,
                result.quality as u8,
                result.file_size,
                result.bits_per_pixel,
                ssim2,
                bfly,
                dssim
            );

            if let Some(ssim2) = result.metrics.ssimulacra2 {
                all_points.push(RDPoint {
                    codec: result.codec_id.clone(),
                    quality_setting: result.quality,
                    bpp: result.bits_per_pixel,
                    quality: ssim2,
                    encode_time_ms: Some(result.encode_time.as_millis() as f64),
                    image: Some(report.name.clone()),
                });
            }
        }
    } else {
        for (i, path) in test_images.iter().enumerate() {
            let name = path.file_stem().unwrap().to_string_lossy();
            println!("\n[{}/{}] Processing: {}", i + 1, test_images.len(), name);

            // Load PNG
            let decoder = png::Decoder::new(fs::File::open(path)?);
            let mut reader = decoder.read_info()?;
            let mut buf = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf)?;
            let bytes = &buf[..info.buffer_size()];

            // Convert to RGB if needed
            let rgb_data = match info.color_type {
                png::ColorType::Rgb => bytes.to_vec(),
                png::ColorType::Rgba => bytes
                    .chunks_exact(4)
                    .flat_map(|c| [c[0], c[1], c[2]])
                    .collect(),
                _ => {
                    println!("  Skipping unsupported color type: {:?}", info.color_type);
                    continue;
                }
            };

            let image = ImageData::RgbSlice {
                data: rgb_data,
                width: info.width as usize,
                height: info.height as usize,
            };

            let report = session.evaluate_image(&name, image)?;
            println!(
                "=== {} ({}x{}) ===",
                report.name, report.width, report.height
            );

            for result in &report.results {
                let dssim = result.metrics.dssim.unwrap_or(0.0);
                let ssim2 = result.metrics.ssimulacra2.unwrap_or(0.0);
                let bfly = result.metrics.butteraugli.unwrap_or(0.0);
                println!(
                    "{:20} Q{:>3}: {:>7} bytes, {:>5.2} bpp, SSIM2={:>5.1}, BA={:>5.2}, DSSIM={:.6}",
                    result.codec_id,
                    result.quality as u8,
                    result.file_size,
                    result.bits_per_pixel,
                    ssim2,
                    bfly,
                    dssim
                );

                if let Some(ssim2) = result.metrics.ssimulacra2 {
                    all_points.push(RDPoint {
                        codec: result.codec_id.clone(),
                        quality_setting: result.quality,
                        bpp: result.bits_per_pixel,
                        quality: ssim2,
                        encode_time_ms: Some(result.encode_time.as_millis() as f64),
                        image: Some(report.name.clone()),
                    });
                }
            }
        }
    }

    // Write results to CSV
    let csv_path = output_dir.join("sharp_yuv_results.csv");
    let mut wtr = csv::Writer::from_path(&csv_path)?;

    wtr.write_record([
        "codec",
        "image",
        "quality",
        "bpp",
        "ssimulacra2",
        "encode_time_ms",
    ])?;

    for point in &all_points {
        wtr.write_record([
            &point.codec,
            point.image.as_deref().unwrap_or(""),
            &point.quality_setting.to_string(),
            &format!("{:.4}", point.bpp),
            &format!("{:.4}", point.quality),
            &format!("{:.1}", point.encode_time_ms.unwrap_or(0.0)),
        ])?;
    }

    wtr.flush()?;
    println!("\nResults written to: {:?}", csv_path);

    // Print summary comparison
    println!("\n=== Summary by Codec ===");
    let codecs = ["mozjpeg-standard", "mozjpeg-yuv-pro", "mozjpeg-sharp-yuv"];

    for codec in codecs {
        let points: Vec<_> = all_points.iter().filter(|p| p.codec == codec).collect();
        if points.is_empty() {
            continue;
        }

        let avg_bpp: f64 = points.iter().map(|p| p.bpp).sum::<f64>() / points.len() as f64;
        let avg_ssim2: f64 = points.iter().map(|p| p.quality).sum::<f64>() / points.len() as f64;

        println!(
            "{:25} avg_bpp={:.3}  avg_ssim2={:.2}",
            codec, avg_bpp, avg_ssim2
        );
    }

    // Compare at specific quality levels
    println!("\n=== Quality 80 Comparison ===");
    let q80_points: Vec<_> = all_points
        .iter()
        .filter(|p| (p.quality_setting - 80.0).abs() < 0.1)
        .collect();

    for codec in codecs {
        let points: Vec<_> = q80_points.iter().filter(|p| p.codec == codec).collect();
        if points.is_empty() {
            continue;
        }

        let avg_bpp: f64 = points.iter().map(|p| p.bpp).sum::<f64>() / points.len() as f64;
        let avg_ssim2: f64 = points.iter().map(|p| p.quality).sum::<f64>() / points.len() as f64;

        println!("{:25} bpp={:.3}  ssim2={:.2}", codec, avg_bpp, avg_ssim2);
    }

    Ok(())
}
