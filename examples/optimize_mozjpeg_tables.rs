//! Simulated annealing optimizer for mozjpeg quantization tables.
//!
//! Compares to jpegli's optimizer by using mozjpeg's direct integer quant tables
//! plus trellis quantization for additional rate-distortion optimization.
//!
//! Key differences from jpegli optimizer:
//! - Simpler search space: 128 u16 values vs 257 f32 values
//! - Direct integer quant tables (what JPEG actually uses)
//! - Trellis quantization provides additional per-block optimization
//! - Progressive encoding for better compression
//!
//! Usage:
//!   cargo run --release --example optimize_mozjpeg_tables -- <corpus_dir> [options]
//!
//! For GPU acceleration (requires CUDA):
//!   CUDA_PATH=/usr/local/cuda-12.6 cargo run --release --example optimize_mozjpeg_tables -- <corpus_dir> --gpu

use mozjpeg_oxide::{Encoder, Preset, Subsampling};
use rayon::prelude::*;
use ssimulacra2::{ColorPrimaries, Rgb, Ssim2Reference, TransferCharacteristic};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// GPU-accelerated SSIM2 support
use cudarse_driver::CuStream;
use cudarse_npp::image::isu::Malloc;
use cudarse_npp::image::{Image, Img, C};
use cudarse_npp::set_stream;
use ssimulacra2_cuda::Ssimulacra2 as GpuSsimulacra2;

// Simple PRNG (xoshiro256++)
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut s = seed;
        let mut state = [0u64; 4];
        for st in state.iter_mut() {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *st = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn gen_range(&mut self, range: std::ops::Range<usize>) -> usize {
        let len = range.end - range.start;
        if len == 0 {
            return range.start;
        }
        range.start + (self.next_u64() as usize % len)
    }

    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gen_i32(&mut self, range: std::ops::Range<i32>) -> i32 {
        let len = (range.end - range.start) as u64;
        if len == 0 {
            return range.start;
        }
        range.start + (self.next_u64() % len) as i32
    }
}

/// ImageMagick/Robidoux luminance quantization table (mozjpeg default)
/// Source: mozjpeg/jcparam.c std_luminance_quant_tbl[3]
const IMAGEMAGICK_LUMA: [u16; 64] = [
    16, 16, 16, 18, 25, 37, 56, 85, // Original DC=16
    16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91, 135, 18, 27, 31, 40, 53, 74, 106,
    156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74, 94, 124, 169, 238, 56, 53, 91, 106, 131,
    169, 226, 311, 85, 75, 135, 156, 189, 238, 311, 418,
];

/// ImageMagick/Robidoux chrominance quantization table (same as luma)
/// Source: mozjpeg/jcparam.c std_chrominance_quant_tbl[3]
const IMAGEMAGICK_CHROMA: [u16; 64] = [
    16, 16, 16, 18, 25, 37, 56, 85, // Original DC=16
    16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91, 135, 18, 27, 31, 40, 53, 74, 106,
    156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74, 94, 124, 169, 238, 56, 53, 91, 106, 131,
    169, 226, 311, 85, 75, 135, 156, 189, 238, 311, 418,
];

/// Optimization state - direct integer quant tables
#[derive(Clone)]
struct OptState {
    /// Luminance quantization table (64 values, 1-255)
    luma: [u16; 64],
    /// Chrominance quantization table (64 values, 1-255)
    chroma: [u16; 64],
}

impl OptState {
    fn new() -> Self {
        Self {
            luma: IMAGEMAGICK_LUMA,
            chroma: IMAGEMAGICK_CHROMA,
        }
    }

    /// Scale tables by quality factor (like libjpeg)
    fn scaled(quality: u8) -> Self {
        let scale = if quality < 50 {
            5000 / quality as u32
        } else {
            200 - 2 * quality as u32
        };

        let mut state = Self::new();
        for i in 0..64 {
            let luma_val = (IMAGEMAGICK_LUMA[i] as u32 * scale + 50) / 100;
            state.luma[i] = luma_val.clamp(1, 255) as u16;

            let chroma_val = (IMAGEMAGICK_CHROMA[i] as u32 * scale + 50) / 100;
            state.chroma[i] = chroma_val.clamp(1, 255) as u16;
        }
        state
    }

    fn to_json(&self) -> String {
        let mut json = String::from("{\n  \"luma\": [");
        for (i, v) in self.luma.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            if i % 8 == 0 {
                json.push_str("\n    ");
            }
            json.push_str(&format!("{}", v));
        }
        json.push_str("\n  ],\n  \"chroma\": [");
        for (i, v) in self.chroma.iter().enumerate() {
            if i > 0 {
                json.push_str(", ");
            }
            if i % 8 == 0 {
                json.push_str("\n    ");
            }
            json.push_str(&format!("{}", v));
        }
        json.push_str("\n  ]\n}\n");
        json
    }

    fn from_json(json: &str) -> Option<Self> {
        let mut state = Self::new();

        if let Some(start) = json.find("\"luma\": [") {
            let start = start + 9;
            if let Some(end) = json[start..].find(']') {
                let values: Vec<u16> = json[start..start + end]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if values.len() == 64 {
                    state.luma.copy_from_slice(&values);
                }
            }
        }

        if let Some(start) = json.find("\"chroma\": [") {
            let start = start + 11;
            if let Some(end) = json[start..].find(']') {
                let values: Vec<u16> = json[start..start + end]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if values.len() == 64 {
                    state.chroma.copy_from_slice(&values);
                }
            }
        }

        Some(state)
    }
}

/// Perturbation types
enum Perturbation {
    /// Single luma value
    SingleLuma { idx: usize, delta: i32 },
    /// Single chroma value
    SingleChroma { idx: usize, delta: i32 },
    /// Block of luma values (frequency band)
    BlockLuma {
        start: usize,
        count: usize,
        delta: i32,
    },
    /// Block of chroma values
    BlockChroma {
        start: usize,
        count: usize,
        delta: i32,
    },
    /// Scale entire luma table
    ScaleLuma { factor: f32 },
    /// Scale entire chroma table
    ScaleChroma { factor: f32 },
    /// Scale DC coefficients specifically
    ScaleDC { delta: i32 },
    /// Scale high-frequency coefficients
    ScaleHighFreq { delta: i32 },
}

fn apply_perturbation(state: &mut OptState, pert: &Perturbation) {
    match pert {
        Perturbation::SingleLuma { idx, delta } => {
            let new_val = state.luma[*idx] as i32 + delta;
            state.luma[*idx] = new_val.clamp(1, 255) as u16;
        }
        Perturbation::SingleChroma { idx, delta } => {
            let new_val = state.chroma[*idx] as i32 + delta;
            state.chroma[*idx] = new_val.clamp(1, 255) as u16;
        }
        Perturbation::BlockLuma {
            start,
            count,
            delta,
        } => {
            for i in *start..(*start + count).min(64) {
                let new_val = state.luma[i] as i32 + delta;
                state.luma[i] = new_val.clamp(1, 255) as u16;
            }
        }
        Perturbation::BlockChroma {
            start,
            count,
            delta,
        } => {
            for i in *start..(*start + count).min(64) {
                let new_val = state.chroma[i] as i32 + delta;
                state.chroma[i] = new_val.clamp(1, 255) as u16;
            }
        }
        Perturbation::ScaleLuma { factor } => {
            for v in state.luma.iter_mut() {
                let new_val = (*v as f32 * factor).round() as i32;
                *v = new_val.clamp(1, 255) as u16;
            }
        }
        Perturbation::ScaleChroma { factor } => {
            for v in state.chroma.iter_mut() {
                let new_val = (*v as f32 * factor).round() as i32;
                *v = new_val.clamp(1, 255) as u16;
            }
        }
        Perturbation::ScaleDC { delta } => {
            // DC is index 0
            let new_luma = state.luma[0] as i32 + delta;
            state.luma[0] = new_luma.clamp(1, 255) as u16;
            let new_chroma = state.chroma[0] as i32 + delta;
            state.chroma[0] = new_chroma.clamp(1, 255) as u16;
        }
        Perturbation::ScaleHighFreq { delta } => {
            // High frequency coefficients (bottom-right quadrant in zigzag)
            for i in 32..64 {
                let new_luma = state.luma[i] as i32 + delta;
                state.luma[i] = new_luma.clamp(1, 255) as u16;
                let new_chroma = state.chroma[i] as i32 + delta;
                state.chroma[i] = new_chroma.clamp(1, 255) as u16;
            }
        }
    }
}

fn random_perturbation(rng: &mut Rng, temperature: f64) -> Perturbation {
    let scale = (temperature.sqrt() * 2.0) as i32 + 1;

    match rng.gen_range(0..100) {
        0..=34 => Perturbation::SingleLuma {
            idx: rng.gen_range(0..64),
            delta: rng.gen_i32(-5 * scale..5 * scale + 1),
        },
        35..=59 => Perturbation::SingleChroma {
            idx: rng.gen_range(0..64),
            delta: rng.gen_i32(-5 * scale..5 * scale + 1),
        },
        60..=69 => Perturbation::BlockLuma {
            start: rng.gen_range(0..64),
            count: rng.gen_range(2..8),
            delta: rng.gen_i32(-3 * scale..3 * scale + 1),
        },
        70..=79 => Perturbation::BlockChroma {
            start: rng.gen_range(0..64),
            count: rng.gen_range(2..8),
            delta: rng.gen_i32(-3 * scale..3 * scale + 1),
        },
        80..=84 => Perturbation::ScaleLuma {
            factor: (1.0 + (rng.gen_f64() - 0.5) * 0.1 * temperature.sqrt()) as f32,
        },
        85..=89 => Perturbation::ScaleChroma {
            factor: (1.0 + (rng.gen_f64() - 0.5) * 0.1 * temperature.sqrt()) as f32,
        },
        90..=94 => Perturbation::ScaleDC {
            delta: rng.gen_i32(-3 * scale..3 * scale + 1),
        },
        _ => Perturbation::ScaleHighFreq {
            delta: rng.gen_i32(-3 * scale..3 * scale + 1),
        },
    }
}

/// Test image with precomputed SSIM2 reference
struct TestImage {
    rgb: Vec<u8>,
    width: usize,
    height: usize,
    name: String,
    ssim2_ref: Ssim2Reference,
}

fn load_png(path: &Path) -> Option<TestImage> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let (width, height) = (info.width as usize, info.height as usize);

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let rgb_frame = Rgb::new(
        rgb.chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .ok()?;
    let ssim2_ref = Ssim2Reference::new(rgb_frame).ok()?;

    Some(TestImage {
        rgb,
        width,
        height,
        name,
        ssim2_ref,
    })
}

fn compute_ssim2_with_ref(
    reference: &Ssim2Reference,
    decoded: &[u8],
    width: usize,
    height: usize,
) -> f64 {
    let decoded_rgb = Rgb::new(
        decoded
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    reference.compare(decoded_rgb).unwrap_or(0.0)
}

/// GPU-accelerated SSIM2 context
struct GpuSsim2Context {
    stream: CuStream,
    tmp_ref: Image<u8, C<3>>,
    tmp_dis: Image<u8, C<3>>,
    ref_linear: Image<f32, C<3>>,
    dis_linear: Image<f32, C<3>>,
    ssimulacra2: GpuSsimulacra2,
}

impl GpuSsim2Context {
    fn new(width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = CuStream::new()?;
        set_stream(stream.inner() as _)?;

        // Allocate GPU buffers
        let tmp_ref: Image<u8, C<3>> = Image::malloc(width, height)?;
        let tmp_dis: Image<u8, C<3>> = tmp_ref.malloc_same_size()?;
        let ref_linear: Image<f32, C<3>> = Image::malloc(width, height)?;
        let dis_linear: Image<f32, C<3>> = ref_linear.malloc_same_size()?;

        // Create ssimulacra2 instance (tied to these dimensions)
        let ssimulacra2 = GpuSsimulacra2::new(&ref_linear, &dis_linear, &stream)?;

        Ok(Self {
            stream,
            tmp_ref,
            tmp_dis,
            ref_linear,
            dis_linear,
            ssimulacra2,
        })
    }

    fn compute(&mut self, reference: &[u8], distorted: &[u8]) -> f64 {
        // compute_from_cpu_srgb_sync handles upload and sRGB->linear conversion
        self.ssimulacra2
            .compute_from_cpu_srgb_sync(
                reference,
                distorted,
                &mut self.tmp_ref,
                &mut self.tmp_dis,
                &mut self.ref_linear,
                &mut self.dis_linear,
                &self.stream,
            )
            .unwrap_or(0.0)
    }
}

/// Initialize CUDA once at startup
fn init_cuda_once() -> bool {
    static INIT: std::sync::Once = std::sync::Once::new();
    static mut SUCCESS: bool = false;

    INIT.call_once(|| unsafe {
        SUCCESS = cudarse_driver::init_cuda_and_primary_ctx().is_ok();
    });

    unsafe { SUCCESS }
}

/// Profiling stats
struct ProfileStats {
    encode_ns: AtomicU64,
    decode_ns: AtomicU64,
    ssim2_ns: AtomicU64,
    count: AtomicU64,
}

impl Default for ProfileStats {
    fn default() -> Self {
        Self {
            encode_ns: AtomicU64::new(0),
            decode_ns: AtomicU64::new(0),
            ssim2_ns: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
}

impl ProfileStats {
    fn add_encode(&self, ns: u64) {
        self.encode_ns.fetch_add(ns, Ordering::Relaxed);
    }
    fn add_decode(&self, ns: u64) {
        self.decode_ns.fetch_add(ns, Ordering::Relaxed);
    }
    fn add_ssim2(&self, ns: u64) {
        self.ssim2_ns.fetch_add(ns, Ordering::Relaxed);
    }
    fn inc_count(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn report(&self) {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return;
        }
        let encode = self.encode_ns.load(Ordering::Relaxed) as f64;
        let decode = self.decode_ns.load(Ordering::Relaxed) as f64;
        let ssim2 = self.ssim2_ns.load(Ordering::Relaxed) as f64;
        let total = encode + decode + ssim2;

        println!(
            "\n=== Hot Loop Profile ({} evaluations, {} threads) ===",
            count,
            rayon::current_num_threads()
        );
        println!(
            "  Encode:  {:>7.2}ms ({:>5.1}%)",
            encode / 1_000_000.0,
            100.0 * encode / total
        );
        println!(
            "  Decode:  {:>7.2}ms ({:>5.1}%)",
            decode / 1_000_000.0,
            100.0 * decode / total
        );
        println!(
            "  SSIM2:   {:>7.2}ms ({:>5.1}%)",
            ssim2 / 1_000_000.0,
            100.0 * ssim2 / total
        );
        println!("  Total CPU time: {:>7.2}ms", total / 1_000_000.0);
        println!(
            "  Per-eval (wall): {:.2}ms",
            total / 1_000_000.0 / count as f64 / rayon::current_num_threads() as f64
        );
    }
}

/// Encode with mozjpeg using custom quant tables + trellis
fn encode_mozjpeg(state: &OptState, rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    // Use quality=50 so custom tables are used exactly as specified (scale factor = 1.0)
    // Quality scaling formula: Q50 → scale=100 (1.0x), Q80 → scale=40 (0.4x)
    // We want no scaling, so Q50 is required.
    Encoder::new(Preset::ProgressiveSmallest)
        .quality(50) // Critical: ensures custom tables used as-is
        .subsampling(Subsampling::S420)
        .custom_luma_qtable(state.luma)
        .custom_chroma_qtable(state.chroma)
        .encode_rgb(rgb, width, height)
        .expect("mozjpeg encode failed")
}

/// Decode JPEG
fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    decoder.decode().ok()
}

fn evaluate_state(state: &OptState, images: &[TestImage], stats: &ProfileStats) -> (f64, usize) {
    let results: Vec<(f64, usize)> = images
        .par_iter()
        .map(|img| {
            let t0 = Instant::now();
            let jpeg = encode_mozjpeg(state, &img.rgb, img.width as u32, img.height as u32);
            stats.add_encode(t0.elapsed().as_nanos() as u64);

            let t1 = Instant::now();
            let decoded = decode_jpeg(&jpeg).expect("decode failed");
            stats.add_decode(t1.elapsed().as_nanos() as u64);

            let t2 = Instant::now();
            let quality_score =
                compute_ssim2_with_ref(&img.ssim2_ref, &decoded, img.width, img.height);
            stats.add_ssim2(t2.elapsed().as_nanos() as u64);

            (quality_score, jpeg.len())
        })
        .collect();

    let (total_quality, total_size) = results
        .iter()
        .fold((0.0, 0), |(q, s), (qi, si)| (q + qi, s + si));

    stats.inc_count();
    (total_quality / images.len() as f64, total_size)
}

fn fitness(ssim2: f64, size: usize, total_pixels: usize, target_bpp: f64) -> f64 {
    let bpp = (size * 8) as f64 / total_pixels as f64;
    // Pareto improvement: maximize SSIM2, but heavily penalize exceeding target bpp
    // Allow slight bpp increase (up to 5%) with moderate penalty
    // Beyond that, heavily penalize
    if bpp <= target_bpp * 1.05 {
        // Within tolerance: just maximize SSIM2, small bonus for smaller size
        ssim2 + 10.0 * (target_bpp - bpp).max(0.0)
    } else {
        // Over budget: heavy penalty
        ssim2 - 500.0 * (bpp - target_bpp * 1.05)
    }
}

fn total_pixels(images: &[TestImage]) -> usize {
    images.iter().map(|img| img.width * img.height).sum()
}

fn load_corpus(corpus_dir: &Path, max_images: usize) -> Vec<TestImage> {
    let mut images = Vec::new();

    // Check if this is the heuristic_outputs/images directory structure
    // where each subdirectory contains an original.png
    let is_heuristic_structure = corpus_dir
        .read_dir()
        .ok()
        .and_then(|mut d| d.next())
        .and_then(|e| e.ok())
        .map(|e| e.path().join("original.png").exists())
        .unwrap_or(false);

    if is_heuristic_structure {
        // Load from heuristic_outputs/images/<hash>/original.png structure
        let mut entries: Vec<_> = fs::read_dir(corpus_dir)
            .expect("Failed to read corpus directory")
            .filter_map(|e| e.ok())
            .collect();

        // Sort for reproducibility
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            if images.len() >= max_images {
                break;
            }

            let original_path = entry.path().join("original.png");
            if original_path.exists() {
                if let Some(img) = load_png(&original_path) {
                    println!("  Loaded: {} ({}x{})", img.name, img.width, img.height);
                    images.push(img);
                }
            }
        }
    } else {
        // Standard flat directory of PNG files
        let entries: Vec<_> = fs::read_dir(corpus_dir)
            .expect("Failed to read corpus directory")
            .filter_map(|e| e.ok())
            .collect();

        for entry in entries {
            if images.len() >= max_images {
                break;
            }

            let path = entry.path();
            if path.extension().map(|e| e == "png").unwrap_or(false) {
                if let Some(img) = load_png(&path) {
                    println!("  Loaded: {} ({}x{})", img.name, img.width, img.height);
                    images.push(img);
                }
            }
        }
    }

    images
}

fn optimize(
    images: &[TestImage],
    quality: u8,
    iterations: usize,
    seed: u64,
    target_bpp: f64,
    checkpoint_path: Option<&Path>,
    initial_state: Option<OptState>,
) -> OptState {
    let mut rng = Rng::new(seed);
    let profile_stats = ProfileStats::default();
    let pixels = total_pixels(images);

    // Initialize with quality-scaled tables as starting point
    let mut current = initial_state.unwrap_or_else(|| OptState::scaled(quality));
    let (current_ssim2, current_size) = evaluate_state(&current, images, &profile_stats);

    // Baseline comparison
    let baseline = OptState::scaled(quality);
    let (baseline_ssim2, baseline_size) = evaluate_state(&baseline, images, &profile_stats);
    let baseline_bpp = (baseline_size * 8) as f64 / pixels as f64;
    println!(
        "\nBaseline (Q{}): SSIM2={:.4}, size={} bytes, bpp={:.3}",
        quality, baseline_ssim2, baseline_size, baseline_bpp
    );
    println!("Target bpp: {:.3}", target_bpp);

    let mut current_fitness = fitness(current_ssim2, current_size, pixels, target_bpp);
    let mut best = current.clone();
    let mut best_fitness = current_fitness;
    let mut best_ssim2 = current_ssim2;
    let mut best_size = current_size;
    let mut best_bpp = (current_size * 8) as f64 / pixels as f64;

    println!(
        "Initial: SSIM2={:.4}, bpp={:.3}, fitness={:.4}",
        current_ssim2, best_bpp, current_fitness
    );

    let initial_temp: f64 = 10.0;
    let final_temp: f64 = 0.001;
    let cooling_rate = (final_temp / initial_temp).powf(1.0 / iterations as f64);

    let mut temperature = initial_temp;
    let mut accepted = 0;
    let mut improved = 0;
    let mut stagnant = 0;
    let reheat_threshold = 200;
    let reheat_temp = 5.0;

    let start = Instant::now();
    let checkpoint_interval = 100;

    for i in 0..iterations {
        let pert = random_perturbation(&mut rng, temperature);
        let mut candidate = current.clone();
        apply_perturbation(&mut candidate, &pert);

        let (cand_ssim2, cand_size) = evaluate_state(&candidate, images, &profile_stats);
        let cand_bpp = (cand_size * 8) as f64 / pixels as f64;
        let cand_fitness = fitness(cand_ssim2, cand_size, pixels, target_bpp);

        let delta = cand_fitness - current_fitness;
        let accept = if delta > 0.0 {
            true
        } else {
            let prob = (delta / temperature).exp();
            rng.gen_f64() < prob
        };

        if accept {
            current = candidate;
            current_fitness = cand_fitness;
            accepted += 1;

            if cand_fitness > best_fitness {
                best = current.clone();
                best_fitness = cand_fitness;
                best_ssim2 = cand_ssim2;
                best_size = cand_size;
                best_bpp = cand_bpp;
                improved += 1;
                stagnant = 0;
            } else {
                stagnant += 1;
            }
        } else {
            stagnant += 1;
        }

        if stagnant >= reheat_threshold && temperature < reheat_temp {
            temperature = reheat_temp;
            stagnant = 0;
            println!(
                "  [Reheating to T={:.2} at iteration {}]",
                temperature,
                i + 1
            );
        }

        temperature *= cooling_rate;

        if (i + 1) % 100 == 0 || i == iterations - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let eta = (iterations - i - 1) as f64 / rate;

            println!(
                "[{:5}/{:5}] T={:.4} SSIM2={:.4} bpp={:.3} (target={:.3}) accept={:.1}% improve={} ETA={:.0}s",
                i + 1,
                iterations,
                temperature,
                best_ssim2,
                best_bpp,
                target_bpp,
                100.0 * accepted as f64 / (i + 1) as f64,
                improved,
                eta
            );
        }

        if let Some(path) = checkpoint_path {
            if (i + 1) % checkpoint_interval == 0 {
                let json = best.to_json();
                let _ = fs::write(path, &json);
            }
        }
    }

    println!("\n=== Optimization Complete ===");
    println!(
        "Best: SSIM2={:.4}, bpp={:.3} (target={:.3})",
        best_ssim2, best_bpp, target_bpp
    );
    println!(
        "vs baseline: SSIM2 {:+.4}, bpp {:+.3}",
        best_ssim2 - baseline_ssim2,
        best_bpp - baseline_bpp
    );
    println!(
        "Accepted: {}/{} ({:.1}%)",
        accepted,
        iterations,
        100.0 * accepted as f64 / iterations as f64
    );
    println!("Improved: {} times", improved);

    profile_stats.report();

    best
}

fn print_usage() {
    eprintln!("Usage: optimize_mozjpeg_tables <corpus_dir> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --quality <N>       Target quality level (default: 85)");
    eprintln!("  --target-bpp <N>    Target bits per pixel (default: baseline bpp)");
    eprintln!("  --iterations <N>    SA iterations (default: 10000)");
    eprintln!("  --max-images <N>    Max images to load (default: 20)");
    eprintln!("  --output <file>     Output file for best tables (JSON)");
    eprintln!("  --resume <file>     Resume from checkpoint");
    eprintln!("  --seed <N>          Random seed (default: 42)");
    eprintln!("  --compare-dc        Compare DC=6 vs DC=16 and exit");
    eprintln!("  --search-low-freq   Grid search low-freq coefficients (DC, H-AC, V-AC)");
    eprintln!("  --gpu               Use GPU-accelerated SSIM2 (requires CUDA)");
    eprintln!();
    eprintln!("GPU mode requires CUDA_PATH to be set:");
    eprintln!("  CUDA_PATH=/usr/local/cuda-12.6 cargo run --release --example optimize_mozjpeg_tables -- ...");
}

/// Compare DC=6 vs DC=16 tables across images
fn compare_dc_values(images: &[TestImage], quality: u8) {
    let profile_stats = ProfileStats::default();

    println!("\n=== DC Coefficient Comparison: DC=6 vs DC=16 ===\n");

    // Create DC=16 baseline (original mozjpeg)
    let dc16 = OptState::scaled(quality);
    // Tables are already scaled, DC=16 is the default

    // Create DC=6 variant
    let mut dc6 = OptState::scaled(quality);
    // At Q50, scale factor is 1.0, so DC=16 becomes 16
    // We want DC=6, so set it directly
    dc6.luma[0] = 6;
    dc6.chroma[0] = 7;

    println!("Testing {} images at Q{}...\n", images.len(), quality);
    println!(
        "{:<14} | {:>8} {:>8} | {:>8} {:>8} | {:>7} {:>7}",
        "Image", "DC16_bpp", "DC16_ss2", "DC6_bpp", "DC6_ss2", "Δbpp%", "Δssim2"
    );
    println!("{}", "-".repeat(80));

    let mut total_dc16_ssim2 = 0.0;
    let mut total_dc6_ssim2 = 0.0;
    let mut total_dc16_size = 0usize;
    let mut total_dc6_size = 0usize;
    let mut total_pixels = 0usize;

    for img in images {
        let pixels = img.width * img.height;
        total_pixels += pixels;

        // Evaluate DC=16
        let (ssim2_16, size_16) = evaluate_state(&dc16, std::slice::from_ref(img), &profile_stats);
        let bpp_16 = (size_16 * 8) as f64 / pixels as f64;

        // Evaluate DC=6
        let (ssim2_6, size_6) = evaluate_state(&dc6, std::slice::from_ref(img), &profile_stats);
        let bpp_6 = (size_6 * 8) as f64 / pixels as f64;

        let delta_bpp_pct = (bpp_6 - bpp_16) / bpp_16 * 100.0;
        let delta_ssim2 = ssim2_6 - ssim2_16;

        total_dc16_ssim2 += ssim2_16;
        total_dc6_ssim2 += ssim2_6;
        total_dc16_size += size_16;
        total_dc6_size += size_6;

        println!(
            "{:<14} | {:>8.3} {:>8.2} | {:>8.3} {:>8.2} | {:>+6.1}% {:>+7.2}",
            &img.name[..img.name.len().min(14)],
            bpp_16,
            ssim2_16,
            bpp_6,
            ssim2_6,
            delta_bpp_pct,
            delta_ssim2
        );
    }

    println!("{}", "-".repeat(80));

    let avg_dc16_ssim2 = total_dc16_ssim2 / images.len() as f64;
    let avg_dc6_ssim2 = total_dc6_ssim2 / images.len() as f64;
    let avg_dc16_bpp = (total_dc16_size * 8) as f64 / total_pixels as f64;
    let avg_dc6_bpp = (total_dc6_size * 8) as f64 / total_pixels as f64;

    println!("\n=== Summary ===");
    println!(
        "DC=16 (mozjpeg default): avg SSIM2={:.2}, avg bpp={:.3}",
        avg_dc16_ssim2, avg_dc16_bpp
    );
    println!(
        "DC=6  (optimized):       avg SSIM2={:.2}, avg bpp={:.3}",
        avg_dc6_ssim2, avg_dc6_bpp
    );
    println!();
    println!(
        "Improvement: {:+.2} SSIM2, {:+.1}% bpp",
        avg_dc6_ssim2 - avg_dc16_ssim2,
        (avg_dc6_bpp - avg_dc16_bpp) / avg_dc16_bpp * 100.0
    );
}

/// Grid search over low-frequency coefficients
/// Tests positions:
/// - (0,0) = DC (index 0) - average brightness
/// - (0,1) = horizontal AC (index 1) - horizontal edges
/// - (1,0) = vertical AC (index 8) - vertical edges
fn search_low_freq_coefficients(images: &[TestImage], quality: u8) {
    let profile_stats = ProfileStats::default();
    let pixels: usize = images.iter().map(|img| img.width * img.height).sum();

    println!("\n=== Low-Frequency Coefficient Grid Search ===\n");
    println!("Testing positions: DC(0,0), H-AC(0,1), V-AC(1,0)");
    println!("Quality: Q{}", quality);
    println!("Images: {}", images.len());
    println!();

    // Get baseline
    let baseline = OptState::scaled(quality);
    let (base_ssim2, base_size) = evaluate_state(&baseline, images, &profile_stats);
    let base_bpp = (base_size * 8) as f64 / pixels as f64;

    println!("Baseline: SSIM2={:.2}, bpp={:.3}", base_ssim2, base_bpp);
    println!(
        "Baseline coefficients: DC={}, H-AC={}, V-AC={}",
        baseline.luma[0], baseline.luma[1], baseline.luma[8]
    );
    println!();

    // Define search ranges relative to baseline
    // DC (index 0): test values from 4 to 20
    // H-AC (index 1): test values from 10 to 24
    // V-AC (index 8): test values from 10 to 24
    let dc_values: Vec<u16> = (4..=14).step_by(2).collect();
    let hac_values: Vec<u16> = (8..=20).step_by(2).collect();
    let vac_values: Vec<u16> = (8..=20).step_by(2).collect();

    println!(
        "Search space: DC={:?}, H-AC={:?}, V-AC={:?}",
        dc_values, hac_values, vac_values
    );
    println!(
        "Total configurations: {}",
        dc_values.len() * hac_values.len() * vac_values.len()
    );
    println!();

    #[derive(Clone)]
    struct Result {
        dc: u16,
        hac: u16,
        vac: u16,
        ssim2: f64,
        bpp: f64,
        delta_ssim2: f64,
        delta_bpp_pct: f64,
    }

    let mut results: Vec<Result> = Vec::new();

    let start = std::time::Instant::now();
    let total = dc_values.len() * hac_values.len() * vac_values.len();
    let mut count = 0;

    for &dc in &dc_values {
        for &hac in &hac_values {
            for &vac in &vac_values {
                let mut state = OptState::scaled(quality);
                // Apply same relative reduction to both luma and chroma
                state.luma[0] = dc;
                state.luma[1] = hac;
                state.luma[8] = vac;
                // Chroma: use similar ratios
                state.chroma[0] = (dc as f32 * 1.1).round() as u16; // slightly higher for chroma
                state.chroma[1] = hac;
                state.chroma[8] = vac;

                let (ssim2, size) = evaluate_state(&state, images, &profile_stats);
                let bpp = (size * 8) as f64 / pixels as f64;

                results.push(Result {
                    dc,
                    hac,
                    vac,
                    ssim2,
                    bpp,
                    delta_ssim2: ssim2 - base_ssim2,
                    delta_bpp_pct: (bpp - base_bpp) / base_bpp * 100.0,
                });

                count += 1;
                if count % 20 == 0 || count == total {
                    let elapsed = start.elapsed().as_secs_f64();
                    let eta = (total - count) as f64 * elapsed / count as f64;
                    print!(
                        "\r[{}/{}] {:.0}s elapsed, ETA {:.0}s   ",
                        count, total, elapsed, eta
                    );
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }
            }
        }
    }
    println!();

    // Sort by SSIM2 (best first)
    results.sort_by(|a, b| b.ssim2.partial_cmp(&a.ssim2).unwrap());

    println!("\n=== Top 10 Configurations by SSIM2 ===");
    println!(
        "{:>4} {:>4} {:>4} | {:>8} {:>8} | {:>8} {:>8}",
        "DC", "H-AC", "V-AC", "SSIM2", "bpp", "ΔSSIM2", "Δbpp%"
    );
    println!("{}", "-".repeat(60));

    for r in results.iter().take(10) {
        println!(
            "{:>4} {:>4} {:>4} | {:>8.2} {:>8.3} | {:>+8.2} {:>+7.1}%",
            r.dc, r.hac, r.vac, r.ssim2, r.bpp, r.delta_ssim2, r.delta_bpp_pct
        );
    }

    // Find Pareto-optimal configurations (maximize SSIM2, minimize bpp)
    println!("\n=== Pareto Front (SSIM2 vs bpp) ===");
    let mut pareto: Vec<&Result> = Vec::new();
    for r in &results {
        let dominated = results.iter().any(|other| {
            other.ssim2 > r.ssim2 && other.bpp <= r.bpp
                || other.ssim2 >= r.ssim2 && other.bpp < r.bpp
        });
        if !dominated {
            pareto.push(r);
        }
    }
    pareto.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).unwrap());

    println!(
        "{:>4} {:>4} {:>4} | {:>8} {:>8} | {:>8} {:>8}",
        "DC", "H-AC", "V-AC", "SSIM2", "bpp", "ΔSSIM2", "Δbpp%"
    );
    println!("{}", "-".repeat(60));
    for r in &pareto {
        println!(
            "{:>4} {:>4} {:>4} | {:>8.2} {:>8.3} | {:>+8.2} {:>+7.1}%",
            r.dc, r.hac, r.vac, r.ssim2, r.bpp, r.delta_ssim2, r.delta_bpp_pct
        );
    }

    // Best configuration at similar bpp to baseline (within 5%)
    println!("\n=== Best at ≤5% bpp increase ===");
    let constrained: Vec<_> = results.iter().filter(|r| r.delta_bpp_pct <= 5.0).collect();
    if let Some(best) = constrained.first() {
        println!("DC={}, H-AC={}, V-AC={}", best.dc, best.hac, best.vac);
        println!(
            "SSIM2={:.2} ({:+.2}), bpp={:.3} ({:+.1}%)",
            best.ssim2, best.delta_ssim2, best.bpp, best.delta_bpp_pct
        );
    }

    profile_stats.report();
}

/// GPU-accelerated grid search over low-frequency coefficients
fn search_low_freq_coefficients_gpu(images: &[TestImage], quality: u8) {
    println!("\n=== Low-Frequency Coefficient Grid Search (GPU) ===\n");

    // Initialize CUDA once
    if !init_cuda_once() {
        eprintln!("Failed to initialize CUDA. Falling back to CPU mode.");
        search_low_freq_coefficients(images, quality);
        return;
    }
    println!("CUDA initialized successfully");

    println!("Testing positions: DC(0,0), H-AC(0,1), V-AC(1,0)");
    println!("Quality: Q{}", quality);
    println!("Images: {}", images.len());
    println!();

    // For GPU, we need to use images of the same size or create contexts per size
    // For simplicity, use only the first image (or filter to same-size images)
    let first_img = &images[0];
    let (width, height) = (first_img.width as u32, first_img.height as u32);

    // Filter to images matching the first image's dimensions
    let matching_images: Vec<_> = images
        .iter()
        .filter(|img| img.width as u32 == width && img.height as u32 == height)
        .collect();

    if matching_images.len() < images.len() {
        println!(
            "Note: Using {} images with dimensions {}x{} (filtered from {})",
            matching_images.len(),
            width,
            height,
            images.len()
        );
    }

    // Create GPU context once for this image size
    let mut gpu_ctx = match GpuSsim2Context::new(width, height) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("GPU init failed: {}. Falling back to CPU mode.", e);
            search_low_freq_coefficients(images, quality);
            return;
        }
    };
    println!("GPU context created for {}x{}", width, height);

    let dc_values: Vec<u16> = (4..=14).step_by(2).collect();
    let hac_values: Vec<u16> = (8..=20).step_by(2).collect();
    let vac_values: Vec<u16> = (8..=20).step_by(2).collect();

    println!(
        "Search space: DC={:?}, H-AC={:?}, V-AC={:?}",
        dc_values, hac_values, vac_values
    );
    let total_configs = dc_values.len() * hac_values.len() * vac_values.len();
    println!("Total configurations: {}", total_configs);
    println!();

    #[derive(Clone)]
    struct SearchResult {
        dc: u16,
        hac: u16,
        vac: u16,
        ssim2: f64,
        bpp: f64,
        delta_ssim2: f64,
        delta_bpp_pct: f64,
    }

    let pixels: usize = matching_images
        .iter()
        .map(|img| img.width * img.height)
        .sum();

    // Get baseline
    let baseline = OptState::scaled(quality);
    let mut base_ssim2_sum = 0.0;
    let mut base_size_sum = 0usize;

    println!("Computing baseline...");
    for img in &matching_images {
        let jpeg = encode_mozjpeg(&baseline, &img.rgb, img.width as u32, img.height as u32);
        base_size_sum += jpeg.len();
        let decoded = decode_jpeg(&jpeg).expect("decode failed");
        let ssim2 = gpu_ctx.compute(&img.rgb, &decoded);
        base_ssim2_sum += ssim2;
    }
    let base_ssim2 = base_ssim2_sum / matching_images.len() as f64;
    let base_bpp = (base_size_sum * 8) as f64 / pixels as f64;

    println!("Baseline: SSIM2={:.2}, bpp={:.3}", base_ssim2, base_bpp);
    println!(
        "Baseline coefficients: DC={}, H-AC={}, V-AC={}",
        baseline.luma[0], baseline.luma[1], baseline.luma[8]
    );
    println!();

    let mut results: Vec<SearchResult> = Vec::new();
    let start = std::time::Instant::now();
    let mut config_count = 0;

    // Process each configuration
    for &dc in &dc_values {
        for &hac in &hac_values {
            for &vac in &vac_values {
                let mut state = OptState::scaled(quality);
                state.luma[0] = dc;
                state.luma[1] = hac;
                state.luma[8] = vac;
                state.chroma[0] = (dc as f32 * 1.1).round() as u16;
                state.chroma[1] = hac;
                state.chroma[8] = vac;

                let mut total_ssim2 = 0.0;
                let mut total_size = 0usize;

                // Process each image with GPU ssimulacra2
                for img in &matching_images {
                    let jpeg =
                        encode_mozjpeg(&state, &img.rgb, img.width as u32, img.height as u32);
                    total_size += jpeg.len();
                    let decoded = decode_jpeg(&jpeg).expect("decode failed");
                    let ssim2 = gpu_ctx.compute(&img.rgb, &decoded);
                    total_ssim2 += ssim2;
                }

                let avg_ssim2 = total_ssim2 / matching_images.len() as f64;
                let bpp = (total_size * 8) as f64 / pixels as f64;

                results.push(SearchResult {
                    dc,
                    hac,
                    vac,
                    ssim2: avg_ssim2,
                    bpp,
                    delta_ssim2: avg_ssim2 - base_ssim2,
                    delta_bpp_pct: (bpp - base_bpp) / base_bpp * 100.0,
                });

                config_count += 1;
                if config_count % 10 == 0 || config_count == total_configs {
                    let elapsed = start.elapsed().as_secs_f64();
                    let eta = (total_configs - config_count) as f64 * elapsed / config_count as f64;
                    print!(
                        "\r[{}/{}] {:.0}s elapsed, ETA {:.0}s   ",
                        config_count, total_configs, elapsed, eta
                    );
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                }
            }
        }
    }
    println!();

    // Sort by SSIM2 (best first)
    results.sort_by(|a, b| b.ssim2.partial_cmp(&a.ssim2).unwrap());

    println!("\n=== Top 10 Configurations by SSIM2 (GPU) ===");
    println!(
        "{:>4} {:>4} {:>4} | {:>8} {:>8} | {:>8} {:>8}",
        "DC", "H-AC", "V-AC", "SSIM2", "bpp", "ΔSSIM2", "Δbpp%"
    );
    println!("{}", "-".repeat(60));

    for r in results.iter().take(10) {
        println!(
            "{:>4} {:>4} {:>4} | {:>8.2} {:>8.3} | {:>+8.2} {:>+7.1}%",
            r.dc, r.hac, r.vac, r.ssim2, r.bpp, r.delta_ssim2, r.delta_bpp_pct
        );
    }

    // Best configuration at similar bpp to baseline (within 5%)
    println!("\n=== Best at ≤5% bpp increase ===");
    let constrained: Vec<_> = results.iter().filter(|r| r.delta_bpp_pct <= 5.0).collect();
    if let Some(best) = constrained.first() {
        println!("DC={}, H-AC={}, V-AC={}", best.dc, best.hac, best.vac);
        println!(
            "SSIM2={:.2} ({:+.2}), bpp={:.3} ({:+.1}%)",
            best.ssim2, best.delta_ssim2, best.bpp, best.delta_bpp_pct
        );
    }

    println!("\nTotal time: {:.1}s", start.elapsed().as_secs_f64());

    // Exit immediately to avoid CUDA cleanup crash (context destruction ordering issue)
    std::process::exit(0);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let corpus_dir = PathBuf::from(&args[1]);
    if !corpus_dir.is_dir() {
        eprintln!("Error: {} is not a directory", corpus_dir.display());
        std::process::exit(1);
    }

    let mut quality: u8 = 85;
    let mut target_bpp: Option<f64> = None;
    let mut iterations: usize = 10000;
    let mut max_images: usize = 20;
    let mut output_path: Option<PathBuf> = None;
    let mut resume_path: Option<PathBuf> = None;
    let mut seed: u64 = 42;
    let mut compare_dc = false;
    let mut search_low_freq = false;
    let mut use_gpu = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--quality" => {
                quality = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(85);
                i += 2;
            }
            "--target-bpp" => {
                target_bpp = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--iterations" => {
                iterations = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10000);
                i += 2;
            }
            "--max-images" => {
                max_images = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(20);
                i += 2;
            }
            "--output" => {
                output_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--resume" => {
                resume_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(42);
                i += 2;
            }
            "--compare-dc" => {
                compare_dc = true;
                i += 1;
            }
            "--search-low-freq" => {
                search_low_freq = true;
                i += 1;
            }
            "--gpu" => {
                use_gpu = true;
                i += 1;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
    }

    println!("=== mozjpeg Quantization Table Optimizer ===");
    println!("Corpus: {}", corpus_dir.display());
    println!("Quality: {}", quality);
    println!("Iterations: {}", iterations);
    println!("Max images: {}", max_images);
    println!("Seed: {}", seed);
    println!();

    println!("Loading images...");
    let images = load_corpus(&corpus_dir, max_images);
    if images.is_empty() {
        eprintln!("Error: No PNG images found in corpus");
        std::process::exit(1);
    }
    println!("Loaded {} images\n", images.len());

    // DC comparison mode - just compare DC=6 vs DC=16 and exit
    if compare_dc {
        compare_dc_values(&images, quality);
        return;
    }

    // Low-frequency grid search mode
    if search_low_freq {
        if use_gpu {
            search_low_freq_coefficients_gpu(&images, quality);
        } else {
            search_low_freq_coefficients(&images, quality);
        }
        return;
    }

    let initial_state = resume_path.as_ref().and_then(|path| {
        println!("Resuming from: {}", path.display());
        fs::read_to_string(path)
            .ok()
            .and_then(|json| OptState::from_json(&json))
    });

    let target_bpp = target_bpp.unwrap_or_else(|| {
        println!("Computing baseline bpp...");
        let baseline = OptState::scaled(quality);
        let pixels = total_pixels(&images);
        let total_size: usize = images
            .iter()
            .map(|img| {
                encode_mozjpeg(&baseline, &img.rgb, img.width as u32, img.height as u32).len()
            })
            .sum();
        let bpp = (total_size * 8) as f64 / pixels as f64;
        println!("Baseline bpp: {:.3}", bpp);
        bpp
    });

    let checkpoint_path = output_path
        .as_ref()
        .map(|p| p.with_extension("checkpoint.json"));

    let best = optimize(
        &images,
        quality,
        iterations,
        seed,
        target_bpp,
        checkpoint_path.as_deref(),
        initial_state,
    );

    if let Some(path) = output_path {
        let json = best.to_json();
        fs::write(&path, &json).expect("Failed to write output");
        println!("\nBest tables saved to: {}", path.display());
    }

    // Print optimized tables
    println!("\n=== Optimized Quantization Tables ===");
    println!("pub const OPTIMIZED_LUMA_QTABLE: [u16; 64] = [");
    for row in 0..8 {
        print!("    ");
        for col in 0..8 {
            print!("{:>3}, ", best.luma[row * 8 + col]);
        }
        println!();
    }
    println!("];");

    println!("\npub const OPTIMIZED_CHROMA_QTABLE: [u16; 64] = [");
    for row in 0..8 {
        print!("    ");
        for col in 0..8 {
            print!("{:>3}, ", best.chroma[row * 8 + col]);
        }
        println!();
    }
    println!("];");
}
