//! Benchmark to discover optimal codec/quality heuristics for the unified quality scale.
//!
//! This benchmark:
//! 1. Encodes each test image at many quality levels with both mozjpeg and jpegli
//! 2. Measures Butteraugli and SSIMULACRA2 for each encoding
//! 3. Builds per-image Pareto fronts
//! 4. Correlates image characteristics with optimal codec choice
//! 5. Outputs heuristic data for the unified quality system
//!
//! Run with:
//! ```
//! cargo run --release --example discover_heuristics -- /path/to/corpus output.json
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// A single encoding data point
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncodingPoint {
    /// Codec used
    codec: String,
    /// Quality value (1-100)
    quality: u8,
    /// File size in bytes
    size: usize,
    /// Bits per pixel
    bpp: f32,
    /// Butteraugli score (lower = better)
    butteraugli: f32,
    /// SSIMULACRA2 score (higher = better)
    ssimulacra2: f32,
    /// Encoding time in milliseconds
    encode_time_ms: u64,
}

/// Image analysis features
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageFeatures {
    /// Image width
    width: usize,
    /// Image height
    height: usize,
    /// Total pixels
    pixels: usize,
    /// Luminance variance
    variance: f32,
    /// Edge density (0-1)
    edge_density: f32,
    /// Chroma complexity (0-1)
    chroma_complexity: f32,
    /// Fraction of uniform 8x8 blocks
    uniform_block_fraction: f32,
    /// Whether image appears to be a photo (vs graphic)
    is_photo: bool,
}

/// Complete benchmark result for one image
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageBenchmark {
    /// Image filename
    filename: String,
    /// Image features
    features: ImageFeatures,
    /// All encoding data points
    points: Vec<EncodingPoint>,
    /// Pareto-optimal points (indices into points)
    pareto_front: Vec<usize>,
    /// Crossover bpp: below this mozjpeg wins, above jpegli wins
    crossover_bpp: Option<f32>,
    /// Best codec at various bpp targets
    codec_at_bpp: HashMap<String, String>,
}

/// Aggregate heuristics discovered from benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiscoveredHeuristics {
    /// Number of images analyzed
    image_count: usize,
    /// Average crossover bpp (where jpegli starts winning)
    avg_crossover_bpp: f32,
    /// Crossover bpp by image type
    crossover_by_type: HashMap<String, f32>,
    /// Quality mapping: unified_q -> (mozjpeg_q, jpegli_q, preferred_codec)
    quality_mapping: Vec<QualityMapEntry>,
    /// Image feature thresholds for codec selection
    feature_thresholds: FeatureThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityMapEntry {
    unified_q: u8,
    target_bpp: f32,
    target_butteraugli: f32,
    mozjpeg_q: u8,
    jpegli_q: u8,
    preferred_codec: String,
    preference_strength: f32, // 0-1, how strongly we prefer this codec
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureThresholds {
    /// Variance threshold above which jpegli is preferred
    variance_jpegli_threshold: f32,
    /// Chroma complexity below which subsampling is safe
    chroma_subsample_threshold: f32,
    /// Edge density threshold for quality-sensitive images
    edge_density_threshold: f32,
}

/// Full benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResults {
    /// Individual image results
    images: Vec<ImageBenchmark>,
    /// Discovered heuristics
    heuristics: DiscoveredHeuristics,
    /// Benchmark metadata
    metadata: BenchmarkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkMetadata {
    /// Timestamp
    timestamp: String,
    /// Quality levels tested
    quality_levels: Vec<u8>,
    /// Codecs tested
    codecs: Vec<String>,
    /// Total encoding time
    total_time_seconds: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <corpus_dir> <output.json>", args[0]);
        eprintln!();
        eprintln!("Environment variables:");
        eprintln!("  MAX_IMAGES=N     Limit to N images");
        eprintln!("  QUALITY_STEP=N   Test every Nth quality level (default: 5)");
        eprintln!("  VERBOSE=1        Show per-image progress");
        std::process::exit(1);
    }

    let corpus_dir = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    // Configuration from environment
    let max_images: usize = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let quality_step: u8 = std::env::var("QUALITY_STEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let verbose = std::env::var("VERBOSE").is_ok();

    // Find PNG images in corpus
    let images = find_images(&corpus_dir, max_images);
    println!("Found {} images in {:?}", images.len(), corpus_dir);

    if images.is_empty() {
        eprintln!("No PNG images found!");
        std::process::exit(1);
    }

    // Quality levels to test
    let quality_levels: Vec<u8> = (1..=100).step_by(quality_step as usize).collect();
    println!("Testing {} quality levels: {:?}...", quality_levels.len(),
             &quality_levels[..quality_levels.len().min(10)]);

    let start = Instant::now();
    let mut image_results = Vec::new();

    for (i, image_path) in images.iter().enumerate() {
        if verbose {
            println!("[{}/{}] Processing {:?}...",
                     i + 1, images.len(), image_path.file_name().unwrap());
        }

        match benchmark_image(image_path, &quality_levels) {
            Ok(result) => {
                if verbose {
                    println!("  {} points, crossover at {:?} bpp",
                             result.points.len(), result.crossover_bpp);
                }
                image_results.push(result);
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
    }

    let total_time = start.elapsed().as_secs_f64();
    println!("\nProcessed {} images in {:.1}s", image_results.len(), total_time);

    // Compute aggregate heuristics
    let heuristics = compute_heuristics(&image_results, &quality_levels);

    // Build final results
    let results = BenchmarkResults {
        images: image_results,
        heuristics,
        metadata: BenchmarkMetadata {
            timestamp: chrono_lite_timestamp(),
            quality_levels,
            codecs: vec!["mozjpeg".to_string(), "jpegli".to_string()],
            total_time_seconds: total_time,
        },
    };

    // Write output
    let json = serde_json::to_string_pretty(&results).unwrap();
    fs::write(&output_path, &json).expect("Failed to write output");
    println!("Results written to {:?}", output_path);

    // Print summary
    print_summary(&results);
}

fn find_images(dir: &Path, max: usize) -> Vec<PathBuf> {
    let mut images = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "png") {
                images.push(path);
                if images.len() >= max {
                    break;
                }
            }
        }
    }

    // Also check subdirectories one level deep
    if images.len() < max {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(subentries) = fs::read_dir(&path) {
                        for subentry in subentries.flatten() {
                            let subpath = subentry.path();
                            if subpath.extension().map_or(false, |e| e == "png") {
                                images.push(subpath);
                                if images.len() >= max {
                                    return images;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    images.sort();
    images
}

fn benchmark_image(
    image_path: &Path,
    quality_levels: &[u8],
) -> Result<ImageBenchmark, String> {
    // Load image
    let file = fs::File::open(image_path)
        .map_err(|e| format!("Failed to open image: {}", e))?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info()
        .map_err(|e| format!("Failed to read PNG info: {}", e))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode PNG: {}", e))?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB if necessary
    let rgb_pixels = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => {
            buf.chunks(4)
                .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
                .collect()
        }
        png::ColorType::Grayscale => {
            buf.iter()
                .flat_map(|&g| [g, g, g])
                .collect()
        }
        png::ColorType::GrayscaleAlpha => {
            buf.chunks(2)
                .flat_map(|ga| [ga[0], ga[0], ga[0]])
                .collect()
        }
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type)),
    };

    // Analyze image features
    let features = analyze_image(&rgb_pixels, width, height);

    // Encode at all quality levels with both codecs
    let mut points = Vec::new();

    for &quality in quality_levels {
        // Mozjpeg encoding
        if let Ok(point) = encode_and_measure_mozjpeg(&rgb_pixels, width, height, quality) {
            points.push(point);
        }

        // Jpegli encoding
        if let Ok(point) = encode_and_measure_jpegli(&rgb_pixels, width, height, quality) {
            points.push(point);
        }
    }

    // Find Pareto front
    let pareto_front = find_pareto_front(&points);

    // Find crossover point
    let crossover_bpp = find_crossover_bpp(&points);

    // Best codec at various bpp targets
    let codec_at_bpp = compute_codec_at_bpp(&points);

    Ok(ImageBenchmark {
        filename: image_path.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        features,
        points,
        pareto_front,
        crossover_bpp,
        codec_at_bpp,
    })
}

fn analyze_image(pixels: &[u8], width: usize, height: usize) -> ImageFeatures {
    // Calculate luminance and stats
    let luma: Vec<f32> = pixels.chunks(3)
        .map(|rgb| 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
        .collect();

    let mean: f32 = luma.iter().sum::<f32>() / luma.len() as f32;
    let variance = luma.iter()
        .map(|&l| (l - mean).powi(2))
        .sum::<f32>() / luma.len() as f32;

    // Edge density
    let mut edge_sum = 0.0f32;
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            if idx + width < luma.len() && idx > 0 {
                let gx = (luma[idx + 1] - luma[idx - 1]).abs();
                let gy = (luma[idx + width] - luma.get(idx.saturating_sub(width)).copied().unwrap_or(0.0)).abs();
                edge_sum += (gx * gx + gy * gy).sqrt();
            }
        }
    }
    let edge_density = edge_sum / ((width.saturating_sub(2)) * (height.saturating_sub(2))) as f32 / 255.0;

    // Chroma complexity
    let mut chroma_var = 0.0f32;
    for rgb in pixels.chunks(3) {
        let cb = -0.169 * rgb[0] as f32 - 0.331 * rgb[1] as f32 + 0.500 * rgb[2] as f32;
        let cr = 0.500 * rgb[0] as f32 - 0.419 * rgb[1] as f32 - 0.081 * rgb[2] as f32;
        chroma_var += cb * cb + cr * cr;
    }
    let chroma_complexity = ((chroma_var / (pixels.len() / 3) as f32).sqrt() / 128.0).min(1.0);

    // Uniform blocks
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let mut uniform_count = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let base_idx = (by * 8 * width + bx * 8) * 3;
            if base_idx + 8 * width * 3 > pixels.len() {
                continue;
            }

            let first = &pixels[base_idx..base_idx + 3];
            let mut is_uniform = true;

            'block: for dy in 0..8 {
                for dx in 0..8 {
                    let idx = base_idx + (dy * width + dx) * 3;
                    if idx + 3 > pixels.len() {
                        break 'block;
                    }
                    if (pixels[idx] as i32 - first[0] as i32).abs() > 4
                        || (pixels[idx + 1] as i32 - first[1] as i32).abs() > 4
                        || (pixels[idx + 2] as i32 - first[2] as i32).abs() > 4
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

    let total_blocks = (blocks_x * blocks_y).max(1);
    let uniform_block_fraction = uniform_count as f32 / total_blocks as f32;

    // Heuristic: is this a photo?
    let is_photo = variance > 500.0
        && chroma_complexity > 0.05
        && uniform_block_fraction < 0.3;

    ImageFeatures {
        width,
        height,
        pixels: width * height,
        variance,
        edge_density,
        chroma_complexity,
        uniform_block_fraction,
        is_photo,
    }
}

fn encode_and_measure_mozjpeg(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<EncodingPoint, String> {
    use mozjpeg_oxide::Encoder;

    let start = Instant::now();

    let encoder = Encoder::new()
        .quality(quality)
        .subsampling(mozjpeg_oxide::Subsampling::S420);

    let jpeg_data = encoder.encode_rgb(pixels, width as u32, height as u32)
        .map_err(|e| format!("mozjpeg encode failed: {:?}", e))?;

    let encode_time = start.elapsed().as_millis() as u64;
    let size = jpeg_data.len();
    let bpp = (size * 8) as f32 / (width * height) as f32;

    // Decode for quality measurement
    let decoded = decode_jpeg(&jpeg_data)?;

    // Measure quality
    let butteraugli = measure_butteraugli(pixels, &decoded, width, height);
    let ssimulacra2 = measure_ssimulacra2(pixels, &decoded, width, height);

    Ok(EncodingPoint {
        codec: "mozjpeg".to_string(),
        quality,
        size,
        bpp,
        butteraugli,
        ssimulacra2,
        encode_time_ms: encode_time,
    })
}

fn encode_and_measure_jpegli(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<EncodingPoint, String> {
    use jpegli::{Encoder, Quality, Subsampling};

    let start = Instant::now();

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(Quality::from_quality(quality as f32))
        .subsampling(Subsampling::S420);

    let jpeg_data = encoder.encode(pixels)
        .map_err(|e| format!("jpegli encode failed: {:?}", e))?;

    let encode_time = start.elapsed().as_millis() as u64;
    let size = jpeg_data.len();
    let bpp = (size * 8) as f32 / (width * height) as f32;

    // Decode for quality measurement
    let decoded = decode_jpeg(&jpeg_data)?;

    // Measure quality
    let butteraugli = measure_butteraugli(pixels, &decoded, width, height);
    let ssimulacra2 = measure_ssimulacra2(pixels, &decoded, width, height);

    Ok(EncodingPoint {
        codec: "jpegli".to_string(),
        quality,
        size,
        bpp,
        butteraugli,
        ssimulacra2,
        encode_time_ms: encode_time,
    })
}

fn decode_jpeg(data: &[u8]) -> Result<Vec<u8>, String> {
    use jpeg_decoder::Decoder;

    let mut decoder = Decoder::new(std::io::Cursor::new(data));
    decoder.decode()
        .map_err(|e| format!("JPEG decode failed: {:?}", e))
}

fn measure_butteraugli(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f32 {
    use codec_eval::metrics::butteraugli::calculate_butteraugli;

    // Handle size mismatch (decoded might have padding)
    let expected_size = width * height * 3;
    if decoded.len() < expected_size {
        return f32::MAX;
    }

    match calculate_butteraugli(original, &decoded[..expected_size], width, height) {
        Ok(score) => score as f32,
        Err(_) => f32::MAX,
    }
}

fn measure_ssimulacra2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f32 {
    use codec_eval::metrics::ssimulacra2::calculate_ssimulacra2;

    let expected_size = width * height * 3;
    if decoded.len() < expected_size {
        return 0.0;
    }

    match calculate_ssimulacra2(original, &decoded[..expected_size], width, height) {
        Ok(score) => score as f32,
        Err(_) => 0.0,
    }
}

fn find_pareto_front(points: &[EncodingPoint]) -> Vec<usize> {
    // A point is Pareto-optimal if no other point is both:
    // - Smaller (lower bpp)
    // - Better quality (lower butteraugli)

    let mut pareto = Vec::new();

    for (i, p) in points.iter().enumerate() {
        let dominated = points.iter().any(|other| {
            other.bpp < p.bpp && other.butteraugli < p.butteraugli
        });

        if !dominated {
            pareto.push(i);
        }
    }

    // Sort by bpp
    pareto.sort_by(|&a, &b| {
        points[a].bpp.partial_cmp(&points[b].bpp).unwrap()
    });

    pareto
}

fn find_crossover_bpp(points: &[EncodingPoint]) -> Option<f32> {
    // Find the bpp where jpegli starts consistently winning on quality

    // Group by bpp buckets
    let mut buckets: HashMap<i32, (Vec<&EncodingPoint>, Vec<&EncodingPoint>)> = HashMap::new();

    for point in points {
        let bucket = (point.bpp * 20.0) as i32; // 0.05 bpp buckets
        let entry = buckets.entry(bucket).or_insert((Vec::new(), Vec::new()));
        if point.codec == "mozjpeg" {
            entry.0.push(point);
        } else {
            entry.1.push(point);
        }
    }

    // Find first bucket where jpegli wins
    let mut sorted_buckets: Vec<_> = buckets.into_iter().collect();
    sorted_buckets.sort_by_key(|&(k, _)| k);

    for (bucket, (moz, jpegli)) in sorted_buckets {
        if moz.is_empty() || jpegli.is_empty() {
            continue;
        }

        let best_moz = moz.iter().map(|p| p.butteraugli).fold(f32::MAX, f32::min);
        let best_jpegli = jpegli.iter().map(|p| p.butteraugli).fold(f32::MAX, f32::min);

        if best_jpegli < best_moz * 0.95 {
            // jpegli is at least 5% better
            return Some(bucket as f32 / 20.0);
        }
    }

    None
}

fn compute_codec_at_bpp(points: &[EncodingPoint]) -> HashMap<String, String> {
    let mut result = HashMap::new();

    for target in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0] {
        let closest: Vec<_> = points.iter()
            .filter(|p| (p.bpp - target).abs() < 0.05)
            .collect();

        if closest.is_empty() {
            continue;
        }

        // Find best quality at this bpp
        let best = closest.iter()
            .min_by(|a, b| a.butteraugli.partial_cmp(&b.butteraugli).unwrap());

        if let Some(p) = best {
            result.insert(format!("{:.2}", target), p.codec.clone());
        }
    }

    result
}

fn compute_heuristics(images: &[ImageBenchmark], quality_levels: &[u8]) -> DiscoveredHeuristics {
    // Average crossover bpp
    let crossovers: Vec<f32> = images.iter()
        .filter_map(|img| img.crossover_bpp)
        .collect();

    let avg_crossover_bpp = if crossovers.is_empty() {
        0.27 // Default fallback
    } else {
        crossovers.iter().sum::<f32>() / crossovers.len() as f32
    };

    // Crossover by image type
    let mut crossover_by_type = HashMap::new();

    let photo_crossovers: Vec<f32> = images.iter()
        .filter(|img| img.features.is_photo)
        .filter_map(|img| img.crossover_bpp)
        .collect();

    if !photo_crossovers.is_empty() {
        crossover_by_type.insert(
            "photo".to_string(),
            photo_crossovers.iter().sum::<f32>() / photo_crossovers.len() as f32,
        );
    }

    let graphic_crossovers: Vec<f32> = images.iter()
        .filter(|img| !img.features.is_photo)
        .filter_map(|img| img.crossover_bpp)
        .collect();

    if !graphic_crossovers.is_empty() {
        crossover_by_type.insert(
            "graphic".to_string(),
            graphic_crossovers.iter().sum::<f32>() / graphic_crossovers.len() as f32,
        );
    }

    // Build quality mapping
    let quality_mapping = build_quality_mapping(images, quality_levels, avg_crossover_bpp);

    // Feature thresholds (simple heuristics based on data)
    let variances: Vec<f32> = images.iter()
        .filter(|img| img.crossover_bpp.map_or(false, |c| c < avg_crossover_bpp))
        .map(|img| img.features.variance)
        .collect();

    let variance_threshold = if variances.is_empty() {
        500.0
    } else {
        variances.iter().sum::<f32>() / variances.len() as f32
    };

    DiscoveredHeuristics {
        image_count: images.len(),
        avg_crossover_bpp,
        crossover_by_type,
        quality_mapping,
        feature_thresholds: FeatureThresholds {
            variance_jpegli_threshold: variance_threshold,
            chroma_subsample_threshold: 0.15,
            edge_density_threshold: 0.1,
        },
    }
}

fn build_quality_mapping(
    images: &[ImageBenchmark],
    _quality_levels: &[u8],
    avg_crossover_bpp: f32,
) -> Vec<QualityMapEntry> {
    let mut mapping = Vec::new();

    // Build mapping for unified quality 0-100 in steps of 5
    for unified_q in (0..=100).step_by(5) {
        // Target bpp: logarithmic mapping from 0.15 to 5.0
        let t = unified_q as f32 / 100.0;
        let target_bpp = 0.15 * (5.0 / 0.15_f32).powf(t);

        // Target butteraugli
        let target_butteraugli = 15.0 * (0.02_f32).powf(t);

        // Find best codec at this bpp
        let preferred_codec = if target_bpp < avg_crossover_bpp {
            "mozjpeg".to_string()
        } else {
            "jpegli".to_string()
        };

        // Find average quality needed for each codec at this bpp
        let (mozjpeg_q, jpegli_q) = estimate_codec_quality_for_bpp(images, target_bpp);

        // Calculate preference strength
        let preference_strength = if target_bpp < 0.20 {
            1.0 // Strong mozjpeg preference at very low bpp
        } else if target_bpp > 0.35 {
            1.0 // Strong jpegli preference at higher bpp
        } else {
            0.5 + 0.5 * ((target_bpp - avg_crossover_bpp) / 0.1).abs().min(1.0)
        };

        mapping.push(QualityMapEntry {
            unified_q: unified_q as u8,
            target_bpp,
            target_butteraugli,
            mozjpeg_q,
            jpegli_q,
            preferred_codec,
            preference_strength,
        });
    }

    mapping
}

fn estimate_codec_quality_for_bpp(images: &[ImageBenchmark], target_bpp: f32) -> (u8, u8) {
    let mut moz_qualities = Vec::new();
    let mut jpegli_qualities = Vec::new();

    for image in images {
        // Find mozjpeg point closest to target bpp
        let moz_points: Vec<_> = image.points.iter()
            .filter(|p| p.codec == "mozjpeg")
            .collect();

        if let Some(closest) = moz_points.iter()
            .min_by(|a, b| (a.bpp - target_bpp).abs().partial_cmp(&(b.bpp - target_bpp).abs()).unwrap())
        {
            if (closest.bpp - target_bpp).abs() < 0.1 {
                moz_qualities.push(closest.quality);
            }
        }

        // Find jpegli point closest to target bpp
        let jpegli_points: Vec<_> = image.points.iter()
            .filter(|p| p.codec == "jpegli")
            .collect();

        if let Some(closest) = jpegli_points.iter()
            .min_by(|a, b| (a.bpp - target_bpp).abs().partial_cmp(&(b.bpp - target_bpp).abs()).unwrap())
        {
            if (closest.bpp - target_bpp).abs() < 0.1 {
                jpegli_qualities.push(closest.quality);
            }
        }
    }

    let mozjpeg_q = if moz_qualities.is_empty() {
        50
    } else {
        (moz_qualities.iter().map(|&q| q as u32).sum::<u32>() / moz_qualities.len() as u32) as u8
    };

    let jpegli_q = if jpegli_qualities.is_empty() {
        50
    } else {
        (jpegli_qualities.iter().map(|&q| q as u32).sum::<u32>() / jpegli_qualities.len() as u32) as u8
    };

    (mozjpeg_q, jpegli_q)
}

fn print_summary(results: &BenchmarkResults) {
    println!("\n=== DISCOVERED HEURISTICS ===\n");

    let h = &results.heuristics;

    println!("Images analyzed: {}", h.image_count);
    println!("Average crossover bpp: {:.3}", h.avg_crossover_bpp);

    println!("\nCrossover by image type:");
    for (typ, bpp) in &h.crossover_by_type {
        println!("  {}: {:.3} bpp", typ, bpp);
    }

    println!("\nFeature thresholds:");
    println!("  Variance (jpegli preferred): > {:.0}", h.feature_thresholds.variance_jpegli_threshold);
    println!("  Chroma (subsample safe): < {:.2}", h.feature_thresholds.chroma_subsample_threshold);

    println!("\nQuality mapping (unified → codec):");
    println!("  {:>4}  {:>6}  {:>6}  {:>6}  {:>6}  {}", "Q", "bpp", "BA", "moz_q", "jpl_q", "codec");
    for entry in &h.quality_mapping {
        println!("  {:>4}  {:>6.2}  {:>6.2}  {:>6}  {:>6}  {}",
                 entry.unified_q,
                 entry.target_bpp,
                 entry.target_butteraugli,
                 entry.mozjpeg_q,
                 entry.jpegli_q,
                 entry.preferred_codec);
    }

    // Pareto summary
    println!("\n=== PARETO FRONT ANALYSIS ===\n");

    let mut moz_wins = 0;
    let mut jpegli_wins = 0;
    let mut ties = 0;

    for image in &results.images {
        for &idx in &image.pareto_front {
            let point = &image.points[idx];
            if point.codec == "mozjpeg" {
                moz_wins += 1;
            } else {
                jpegli_wins += 1;
            }
        }
    }

    println!("Pareto front composition:");
    println!("  mozjpeg points: {}", moz_wins);
    println!("  jpegli points: {}", jpegli_wins);
    println!("  mozjpeg share: {:.1}%", 100.0 * moz_wins as f64 / (moz_wins + jpegli_wins) as f64);
}

fn chrono_lite_timestamp() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}
