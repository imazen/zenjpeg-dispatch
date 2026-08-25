//! Benchmark comparing Harvard simulated annealing quantization tables
//! against standard mozjpeg and jpegli tables.
//!
//! Usage:
//!   cargo run --release --example sa_tables_benchmark

use codec_eval::{
    decode::jpeg_decode_callback, EvalConfig, EvalSession, ImageData, MetricConfig,
    ViewingCondition,
};
use mozjpeg_oxide::Preset;
use std::fs;
use std::path::PathBuf;
use zenjpeg_dispatch::sa_tables::{
    get_interpolated_sa_table, select_sa_table, select_sa_table_compress, SA_LUMA_Q35,
    SA_LUMA_Q35_COMPRESS, SA_LUMA_Q50, SA_LUMA_Q50_COMPRESS, SA_LUMA_Q75, SA_LUMA_Q75_COMPRESS,
    SA_LUMA_Q95,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = PathBuf::from("comparison_outputs/sa_tables");
    fs::create_dir_all(&output_dir)?;

    let metrics = MetricConfig::perceptual();

    // Test at the 4 SA trained quality points plus some intermediates
    // to see how "nearest" selection performs at non-exact points
    let quality_levels: Vec<f64> = vec![
        35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
    ];

    let config = EvalConfig::builder()
        .report_dir(&output_dir)
        .viewing(ViewingCondition::desktop())
        .metrics(metrics)
        .quality_levels(quality_levels.clone())
        .build();

    let mut session = EvalSession::new(config);

    // Register mozjpeg-oxide with standard tables
    session.add_codec_with_decode(
        "mozjpeg-std",
        "0.1.0",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;

            let encoder = mozjpeg_oxide::Encoder::new(Preset::BaselineBalanced)
                .quality(quality)
                .subsampling(mozjpeg_oxide::Subsampling::S420);
            encoder
                .encode_rgb(&rgb, width, height)
                .map_err(|e| codec_eval::Error::Codec {
                    codec: "mozjpeg-std".to_string(),
                    message: e.to_string(),
                })
        }),
        jpeg_decode_callback(),
    );

    // Register mozjpeg-oxide with SA tables - NEAREST selection (no interpolation)
    // Picks the closest trained table (Q35, Q50, Q75, or Q95)
    session.add_codec_with_decode(
        "mozjpeg-sa-nearest",
        "0.1.0",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;

            // Get the nearest SA table (no interpolation)
            let sa_luma = *select_sa_table(quality);

            // Use Q50 to prevent any scaling (scale_factor = 100 = 1:1)
            let encoder = mozjpeg_oxide::Encoder::new(Preset::BaselineBalanced)
                .quality(50) // Q50 = no scaling on luma
                .custom_luma_qtable(sa_luma)
                .subsampling(mozjpeg_oxide::Subsampling::S420);
            encoder
                .encode_rgb(&rgb, width, height)
                .map_err(|e| codec_eval::Error::Codec {
                    codec: "mozjpeg-sa-nearest".to_string(),
                    message: e.to_string(),
                })
        }),
        jpeg_decode_callback(),
    );

    // Register mozjpeg-oxide with SA tables - INTERPOLATED between trained points
    session.add_codec_with_decode(
        "mozjpeg-sa-interp",
        "0.1.0",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;

            // Get interpolated SA table
            let sa_luma = get_interpolated_sa_table(quality);

            // Use Q50 to prevent any scaling
            let encoder = mozjpeg_oxide::Encoder::new(Preset::BaselineBalanced)
                .quality(50)
                .custom_luma_qtable(sa_luma)
                .subsampling(mozjpeg_oxide::Subsampling::S420);
            encoder
                .encode_rgb(&rgb, width, height)
                .map_err(|e| codec_eval::Error::Codec {
                    codec: "mozjpeg-sa-interp".to_string(),
                    message: e.to_string(),
                })
        }),
        jpeg_decode_callback(),
    );

    // Register jpegli for reference
    session.add_codec_with_decode(
        "jpegli",
        "0.1.0",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as f32;

            let encoder = jpegli::Encoder::new()
                .width(width)
                .height(height)
                .pixel_format(jpegli::PixelFormat::Rgb)
                .jpegli_quality(jpegli::Quality::Traditional(quality));
            encoder.encode(&rgb).map_err(|e| codec_eval::Error::Codec {
                codec: "jpegli".to_string(),
                message: e.to_string(),
            })
        }),
        jpeg_decode_callback(),
    );

    println!("Registered {} codecs", session.codec_count());
    println!("Testing quality levels: {:?}", quality_levels);

    // Find test images
    let corpus_dir = PathBuf::from("../codec-eval/codec-corpus/CID22/CID22-512/training");
    let test_images: Vec<PathBuf> = if corpus_dir.exists() {
        fs::read_dir(&corpus_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "png"))
            .take(8) // Use 8 images for reasonably comprehensive benchmark
            .collect()
    } else {
        println!("CID22 corpus not found at {:?}", corpus_dir);
        vec![]
    };

    if test_images.is_empty() {
        eprintln!("No test images found!");
        return Ok(());
    }

    println!("\nProcessing {} images...\n", test_images.len());

    // Aggregate results
    let mut all_results: Vec<(String, u8, f64, f64, f64, f64)> = Vec::new(); // (codec, q, bpp, ssim2, ba, dssim)

    for img_path in &test_images {
        let img_name = img_path.file_stem().unwrap().to_string_lossy();

        // Load PNG
        let decoder = png::Decoder::new(fs::File::open(img_path)?);
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
            _ => continue,
        };

        let image = ImageData::RgbSlice {
            data: rgb_data,
            width: info.width as usize,
            height: info.height as usize,
        };

        let report = session.evaluate_image(&img_name, image)?;

        for result in &report.results {
            let ssim2 = result.metrics.ssimulacra2.unwrap_or(0.0);
            let ba = result.metrics.butteraugli.unwrap_or(999.0);
            let dssim = result.metrics.dssim.unwrap_or(999.0);

            all_results.push((
                result.codec_id.clone(),
                result.quality as u8,
                result.bits_per_pixel,
                ssim2,
                ba,
                dssim,
            ));
        }

        session.write_image_report(&report)?;
    }

    // Compute averages per codec per quality
    println!("\n=== Aggregated Results (Averages) ===\n");
    println!(
        "{:20} {:>5} {:>8} {:>8} {:>8} {:>10}",
        "Codec", "Q", "BPP", "SSIM2", "BA", "DSSIM"
    );
    println!("{:-<70}", "");

    let codecs: Vec<String> = [
        "mozjpeg-std",
        "mozjpeg-sa-nearest",
        "mozjpeg-sa-interp",
        "jpegli",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let mut csv_output = String::from("codec,quality,bpp,ssimulacra2,butteraugli,dssim\n");

    for q in &quality_levels {
        let q_u8 = *q as u8;
        for codec in &codecs {
            let matching: Vec<_> = all_results
                .iter()
                .filter(|(c, qv, _, _, _, _)| c == codec && *qv == q_u8)
                .collect();

            if matching.is_empty() {
                continue;
            }

            let n = matching.len() as f64;
            let avg_bpp: f64 = matching.iter().map(|(_, _, b, _, _, _)| b).sum::<f64>() / n;
            let avg_ssim2: f64 = matching.iter().map(|(_, _, _, s, _, _)| s).sum::<f64>() / n;
            let avg_ba: f64 = matching.iter().map(|(_, _, _, _, b, _)| b).sum::<f64>() / n;
            let avg_dssim: f64 = matching.iter().map(|(_, _, _, _, _, d)| d).sum::<f64>() / n;

            println!(
                "{:20} {:>5} {:>8.3} {:>8.2} {:>8.3} {:>10.6}",
                codec, q_u8, avg_bpp, avg_ssim2, avg_ba, avg_dssim
            );

            csv_output.push_str(&format!(
                "{},{},{:.4},{:.2},{:.4},{:.8}\n",
                codec, q_u8, avg_bpp, avg_ssim2, avg_ba, avg_dssim
            ));
        }
        println!();
    }

    // Write CSV
    fs::write(output_dir.join("sa_comparison.csv"), csv_output)?;
    println!("\nResults written to {:?}", output_dir);

    // Summary analysis
    println!("\n=== Summary: SA Tables vs Standard ===\n");

    // Compare at key quality points
    for target_q in [35, 50, 75, 95] {
        let std_results: Vec<_> = all_results
            .iter()
            .filter(|(c, q, _, _, _, _)| c == "mozjpeg-std" && *q == target_q)
            .collect();
        let sa_results: Vec<_> = all_results
            .iter()
            .filter(|(c, q, _, _, _, _)| c == "mozjpeg-sa-nearest" && *q == target_q)
            .collect();
        let jpegli_results: Vec<_> = all_results
            .iter()
            .filter(|(c, q, _, _, _, _)| c == "jpegli" && *q == target_q)
            .collect();

        if std_results.is_empty() || sa_results.is_empty() {
            continue;
        }

        let n = std_results.len() as f64;
        let std_bpp: f64 = std_results.iter().map(|(_, _, b, _, _, _)| b).sum::<f64>() / n;
        let sa_bpp: f64 = sa_results.iter().map(|(_, _, b, _, _, _)| b).sum::<f64>() / n;
        let jpegli_bpp: f64 = if !jpegli_results.is_empty() {
            jpegli_results
                .iter()
                .map(|(_, _, b, _, _, _)| b)
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        let std_ssim2: f64 = std_results.iter().map(|(_, _, _, s, _, _)| s).sum::<f64>() / n;
        let sa_ssim2: f64 = sa_results.iter().map(|(_, _, _, s, _, _)| s).sum::<f64>() / n;

        let bpp_reduction = (1.0 - sa_bpp / std_bpp) * 100.0;
        let ssim2_diff = sa_ssim2 - std_ssim2;

        println!("Q{}: SA vs Standard:", target_q);
        println!(
            "  BPP:   {:.3} -> {:.3} ({:+.1}%)",
            std_bpp, sa_bpp, -bpp_reduction
        );
        println!(
            "  SSIM2: {:.2} -> {:.2} ({:+.2})",
            std_ssim2, sa_ssim2, ssim2_diff
        );
        if jpegli_bpp > 0.0 {
            println!("  (jpegli: {:.3} bpp for reference)", jpegli_bpp);
        }
        println!();
    }

    Ok(())
}
