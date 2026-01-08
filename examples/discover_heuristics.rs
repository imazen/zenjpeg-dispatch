//! Benchmark infrastructure for discovering optimal codec/quality heuristics.
//!
//! Features:
//! - Incremental caching with atomic writes
//! - Per-config cache invalidation via code hashing
//! - BPP-bounded bidirectional quality iteration
//! - Human-readable filenames sorted by BPP
//! - Master CSV with all results + image analysis
//! - Resumable across runs and corpora
//!
//! Run with:
//! ```
//! cargo run --release --example discover_heuristics -- \
//!   --corpus /path/to/images \
//!   --output ./benchmark_cache \
//!   --min-bpp 0.2 --max-bpp 2.0 \
//!   --step 5
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use chrono::{DateTime, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ============================================================================
// CLI Arguments
// ============================================================================

// ============================================================================
// Subsampling Configuration
// ============================================================================

/// Chroma subsampling mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum Subsampling {
    /// 4:2:0 - Quarter chroma resolution (most compression)
    S420,
    /// 4:2:2 - Half horizontal chroma resolution
    S422,
    /// 4:4:4 - Full chroma resolution (best quality)
    S444,
    /// Use evalchroma crate to decide based on image content
    Auto,
}

impl Subsampling {
    fn as_str(&self) -> &'static str {
        match self {
            Subsampling::S420 => "420",
            Subsampling::S422 => "422",
            Subsampling::S444 => "444",
            Subsampling::Auto => "auto",
        }
    }
}

impl std::fmt::Display for Subsampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Encoder Configuration (nested enum)
// ============================================================================

/// Encoder configuration with encoder-specific options.
/// Each variant represents a distinct encoder with its settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum Config {
    /// Mozilla's optimized JPEG encoder (baseline mode) - Rust port
    MozJpeg { subsampling: Subsampling },
    /// Mozilla's max compression (progressive + optimize_scans) - Rust port
    MozJpegMax { subsampling: Subsampling },
    /// C mozjpeg reference implementation (via mozjpeg crate)
    CMozJpeg { subsampling: Subsampling },
    /// C mozjpeg max compression (progressive + optimize_scans)
    CMozJpegMax { subsampling: Subsampling },
    /// Google's perceptual JPEG encoder
    Jpegli { subsampling: Subsampling },
    /// Jpegli with XYB color space (more perceptually optimized)
    JpegliXyb { subsampling: Subsampling },
    /// Zenjpeg hybrid encoder (combines mozjpeg trellis + jpegli AQ)
    Zenjpeg { subsampling: Subsampling },
}

/// Trait for encoding images with a configuration
trait Encode {
    /// Get the string key for this config (used in filenames and cache)
    fn key(&self) -> String;

    /// Get the source files that affect this config's behavior (for cache invalidation)
    fn source_files(&self) -> Vec<&'static str>;

    /// Encode RGB pixels to JPEG
    fn encode(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        quality: u8,
    ) -> Result<Vec<u8>, String>;
}

impl Encode for Config {
    fn key(&self) -> String {
        match self {
            Config::MozJpeg { subsampling } => format!("mozjpeg-{}", subsampling),
            Config::MozJpegMax { subsampling } => format!("mozjpeg-max-{}", subsampling),
            Config::CMozJpeg { subsampling } => format!("cmozjpeg-{}", subsampling),
            Config::CMozJpegMax { subsampling } => format!("cmozjpeg-max-{}", subsampling),
            Config::Jpegli { subsampling } => format!("jpegli-{}", subsampling),
            Config::JpegliXyb { subsampling } => format!("jpegli-xyb-{}", subsampling),
            Config::Zenjpeg { subsampling } => format!("zenjpeg-{}", subsampling),
        }
    }

    fn source_files(&self) -> Vec<&'static str> {
        self.source_dirs()
    }

    fn encode(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        quality: u8,
    ) -> Result<Vec<u8>, String> {
        match self {
            Config::MozJpeg { subsampling } => {
                encode_mozjpeg(pixels, width, height, quality, *subsampling, false)
            }
            Config::MozJpegMax { subsampling } => {
                encode_mozjpeg(pixels, width, height, quality, *subsampling, true)
            }
            Config::CMozJpeg { subsampling } => {
                encode_cmozjpeg(pixels, width, height, quality, *subsampling, false)
            }
            Config::CMozJpegMax { subsampling } => {
                encode_cmozjpeg(pixels, width, height, quality, *subsampling, true)
            }
            Config::Jpegli { subsampling } => {
                encode_jpegli(pixels, width, height, quality, *subsampling, false)
            }
            Config::JpegliXyb { subsampling } => {
                encode_jpegli(pixels, width, height, quality, *subsampling, true)
            }
            Config::Zenjpeg { subsampling } => {
                encode_zenjpeg(pixels, width, height, quality, *subsampling)
            }
        }
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.key())
    }
}

/// Version info for cache invalidation tracking.
/// When encoding logic changes, bump version and record the old hash/commit.
#[derive(Debug, Clone)]
struct VersionInfo {
    version: u32,
    /// Hash from the PREVIOUS version (for documentation/audit trail)
    old_hash: &'static str,
    /// Commit from the PREVIOUS version (for documentation/audit trail)
    old_commit: &'static str,
}

impl Config {
    // =========================================================================
    // Config Sets
    // =========================================================================

    /// Minimal baseline configs - one per encoder, 4:2:0 only
    fn baseline() -> Vec<Config> {
        vec![
            Config::MozJpeg {
                subsampling: Subsampling::S420,
            },
            Config::Jpegli {
                subsampling: Subsampling::S420,
            },
        ]
    }

    /// Test subset - configs we actively benchmark
    fn test_subset() -> Vec<Config> {
        vec![
            Config::MozJpeg {
                subsampling: Subsampling::S420,
            },
            Config::MozJpeg {
                subsampling: Subsampling::S444,
            },
            Config::MozJpegMax {
                subsampling: Subsampling::S420,
            },
            Config::MozJpegMax {
                subsampling: Subsampling::S444,
            },
            Config::CMozJpeg {
                subsampling: Subsampling::S420,
            },
            Config::CMozJpegMax {
                subsampling: Subsampling::S420,
            },
            Config::Jpegli {
                subsampling: Subsampling::S420,
            },
            Config::Jpegli {
                subsampling: Subsampling::S444,
            },
        ]
    }

    /// All possible configs (including experimental/future)
    fn all() -> Vec<Config> {
        vec![
            Config::MozJpeg {
                subsampling: Subsampling::S420,
            },
            Config::MozJpeg {
                subsampling: Subsampling::S422,
            },
            Config::MozJpeg {
                subsampling: Subsampling::S444,
            },
            Config::Jpegli {
                subsampling: Subsampling::S420,
            },
            Config::Jpegli {
                subsampling: Subsampling::S422,
            },
            Config::Jpegli {
                subsampling: Subsampling::S444,
            },
            // Future:
            // Config::JpegliXyb { subsampling: Subsampling::S444 },
            // Config::Zenjpeg { subsampling: Subsampling::Auto },
        ]
    }

    // =========================================================================
    // Cache Invalidation Info
    // =========================================================================

    /// Directories/files to hash for cache invalidation.
    /// When any of these change, the cache for this config is invalidated.
    fn source_dirs(&self) -> Vec<&'static str> {
        match self {
            Config::MozJpeg { .. } => vec!["examples/discover_heuristics.rs"],
            Config::MozJpegMax { .. } => vec!["examples/discover_heuristics.rs"],
            Config::CMozJpeg { .. } => vec!["examples/discover_heuristics.rs"],
            Config::CMozJpegMax { .. } => vec!["examples/discover_heuristics.rs"],
            Config::Jpegli { .. } => vec!["examples/discover_heuristics.rs"],
            Config::JpegliXyb { .. } => vec!["examples/discover_heuristics.rs"],
            Config::Zenjpeg { .. } => vec![
                "examples/discover_heuristics.rs",
                "src/", // Zenjpeg uses our own encoder
            ],
        }
    }

    /// Version info for this config.
    /// When you change encoding logic:
    /// 1. Run benchmark - it will error with old/new hash and commit
    /// 2. Increment version and paste old_hash/old_commit from error message
    fn version_info(&self) -> VersionInfo {
        // Version history:
        // v3: Added MozJpegMax. Main dataset with 100k files.
        // v4: mozjpeg-oxide API change (Encoder::new -> baseline_optimized)
        // v5: jpegli-rs encoder output changed.
        match self {
            // ----------------------------------------------------------------
            // MozJpeg configs - v5: API changed to baseline_optimized()
            // (v4 files were created with old Encoder::new() API)
            // ----------------------------------------------------------------
            Config::MozJpeg { .. } => VersionInfo {
                version: 5,
                old_hash: "sha256:013018d04a91f977",
                old_commit: "e04ea4db6538c6c4ba59fb04a38b8d29941704ec",
            },

            // ----------------------------------------------------------------
            // MozJpegMax configs - v5: API changed
            // ----------------------------------------------------------------
            Config::MozJpegMax { .. } => VersionInfo {
                version: 5,
                old_hash: "sha256:013018d04a91f977",
                old_commit: "",
            },

            // ----------------------------------------------------------------
            // C mozjpeg configs (reference implementation) - unchanged
            // ----------------------------------------------------------------
            Config::CMozJpeg { .. } => VersionInfo {
                version: 3,
                old_hash: "sha256:013018d04a91f977",
                old_commit: "",
            },
            Config::CMozJpegMax { .. } => VersionInfo {
                version: 3,
                old_hash: "sha256:013018d04a91f977",
                old_commit: "",
            },

            // ----------------------------------------------------------------
            // Jpegli configs - v5: encoder output changed since v3/v4
            // ----------------------------------------------------------------
            Config::Jpegli { .. } => VersionInfo {
                version: 5,
                old_hash: "sha256:013018d04a91f977",
                old_commit: "e04ea4db6538c6c4ba59fb04a38b8d29941704ec",
            },

            // ----------------------------------------------------------------
            // JpegliXyb configs - v5 with jpegli
            // ----------------------------------------------------------------
            Config::JpegliXyb { .. } => VersionInfo {
                version: 5,
                old_hash: "sha256:013018d04a91f977",
                old_commit: "e04ea4db6538c6c4ba59fb04a38b8d29941704ec",
            },

            // ----------------------------------------------------------------
            // Zenjpeg configs (experimental)
            // ----------------------------------------------------------------
            Config::Zenjpeg { .. } => VersionInfo {
                version: 3,
                old_hash: "sha256:013018d04a91f977",
                old_commit: "e04ea4db6538c6c4ba59fb04a38b8d29941704ec",
            },
        }
    }
}

/// If true, trust existing cache even if code hash changed.
/// Set to true temporarily if you made non-functional changes (comments, formatting).
/// WARNING: Setting this permanently defeats cache invalidation!
const ASSERT_UNCHANGED: bool = true; // Keep true until code is committed

#[derive(Parser, Debug)]
#[command(name = "discover_heuristics")]
#[command(about = "Benchmark codec configurations to discover optimal heuristics")]
struct Args {
    /// Path to corpus directory containing PNG images
    #[arg(long)]
    corpus: PathBuf,

    /// Output directory for cache and results
    #[arg(long)]
    output: PathBuf,

    /// Minimum BPP to test (stop iterating when below this)
    #[arg(long, default_value = "0.15")]
    min_bpp: f32,

    /// Maximum BPP to test (stop iterating when above this)
    #[arg(long, default_value = "3.0")]
    max_bpp: f32,

    /// Quality step size (quality = 100 - step * n)
    #[arg(long, default_value = "1")]
    step: u8,

    /// Force re-encode all (ignore cache completely)
    #[arg(long)]
    force: bool,

    /// Maximum images to process (for testing)
    #[arg(long)]
    max_images: Option<usize>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Run full verification: compare all cached encodings against current codec output
    #[arg(long)]
    verify: bool,

    /// Skip the startup quick-check (3 quality levels, 1 image per config)
    #[arg(long)]
    skip_verify: bool,

    /// Use GPU-accelerated SSIMULACRA2 (requires --features gpu and CUDA)
    #[arg(long)]
    gpu: bool,
}

// ============================================================================
// Data Structures
// ============================================================================

/// Image analysis results (independent of encoding decisions)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageAnalysis {
    source_hash: String,
    source_name: String,
    width: usize,
    height: usize,
    pixels: usize,
    variance: f32,
    edge_density: f32,
    chroma_complexity: f32,
    uniform_block_fraction: f32,
    has_high_frequency: bool,
    color_count_estimate: u32,
    timestamp: DateTime<Utc>,
}

/// Per-encoding metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncodingMetrics {
    source_hash: String,
    config_key: String,
    quality: u8,
    cache_version: u32,
    size_bytes: usize,
    bpp: f32,
    butteraugli: f32,
    ssimulacra2: f32,
    dssim: f32,
    encode_time_ms: u64,
    timestamp: DateTime<Utc>,
}

/// Cache manifest storing per-config versions and code hashes
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheManifest {
    /// Global hash of all source files (for reference)
    global_code_hash: String,
    /// Git commit hash of the repo (if available)
    #[serde(default)]
    git_commit: Option<String>,
    /// Per-config version and hash
    configs: HashMap<String, ConfigCacheEntry>,
    /// Last updated
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigCacheEntry {
    version: u32,
    code_hash: String,
    #[serde(default)]
    git_commit: Option<String>,
    source_files: Vec<String>,
}

/// CSV row for master results file
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CsvRow {
    source_hash: String,
    source_name: String,
    width: usize,
    height: usize,
    variance: f32,
    edge_density: f32,
    chroma_complexity: f32,
    uniform_block_fraction: f32,
    config_key: String,
    quality: u8,
    cache_version: u32,
    size_bytes: usize,
    bpp: f32,
    butteraugli: f32,
    ssimulacra2: f32,
    dssim: f32,
    encode_time_ms: u64,
    timestamp: String,
}

/// Thread-safe run statistics using atomics
struct AtomicRunStats {
    images_processed: AtomicUsize,
    images_skipped: AtomicUsize,
    encodings_performed: AtomicUsize,
    encodings_cached: AtomicUsize,
    total_encode_time_ms: AtomicU64,
    total_metric_time_ms: AtomicU64,
    // Per-metric timing breakdown
    total_butteraugli_ms: AtomicU64,
    total_ssim2_ms: AtomicU64,
    total_dssim_ms: AtomicU64,
    total_decode_ms: AtomicU64,
    errors: Mutex<Vec<String>>,
}

impl AtomicRunStats {
    fn new() -> Self {
        Self {
            images_processed: AtomicUsize::new(0),
            images_skipped: AtomicUsize::new(0),
            encodings_performed: AtomicUsize::new(0),
            encodings_cached: AtomicUsize::new(0),
            total_encode_time_ms: AtomicU64::new(0),
            total_metric_time_ms: AtomicU64::new(0),
            total_butteraugli_ms: AtomicU64::new(0),
            total_ssim2_ms: AtomicU64::new(0),
            total_dssim_ms: AtomicU64::new(0),
            total_decode_ms: AtomicU64::new(0),
            errors: Mutex::new(Vec::new()),
        }
    }

    fn add_error(&self, error: String) {
        self.errors.lock().unwrap().push(error);
    }

    fn print_timing_breakdown(&self) {
        let encode = self.total_encode_time_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        let decode = self.total_decode_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        let butteraugli = self.total_butteraugli_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        let ssim2 = self.total_ssim2_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        let dssim = self.total_dssim_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        let total = encode + decode + butteraugli + ssim2 + dssim;

        println!("\n{:=^70}", " TIMING BREAKDOWN ");
        println!("{:<20} {:>10.1}s  ({:>5.1}%)", "Encoding:", encode, 100.0 * encode / total);
        println!("{:<20} {:>10.1}s  ({:>5.1}%)", "Decoding:", decode, 100.0 * decode / total);
        println!("{:<20} {:>10.1}s  ({:>5.1}%)", "Butteraugli:", butteraugli, 100.0 * butteraugli / total);
        println!("{:<20} {:>10.1}s  ({:>5.1}%)", "SSIMULACRA2:", ssim2, 100.0 * ssim2 / total);
        println!("{:<20} {:>10.1}s  ({:>5.1}%)", "DSSIM:", dssim, 100.0 * dssim / total);
        println!("{:-<70}", "");
        println!("{:<20} {:>10.1}s", "Total measured:", total);
    }
}

/// Work item for parallel processing
#[derive(Clone)]
struct WorkItem {
    image_path: PathBuf,
    rgb_pixels: Arc<Vec<u8>>,
    width: usize,
    height: usize,
    config: Config,
    quality: u8,
    analysis: Arc<ImageAnalysis>,
    image_dir: PathBuf,
    cache_version: u32,
}

/// Result from processing a work item
struct WorkResult {
    analysis: Arc<ImageAnalysis>,
    metrics: Option<EncodingMetrics>,
    cached: bool,
    error: Option<String>,
}

// ============================================================================
// Code Hashing
// ============================================================================

fn compute_file_hash(path: &Path) -> Result<String, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(content.as_bytes());
    Ok(hex::encode(hasher.finalize())[..16].to_string())
}

fn compute_config_code_hash(source_files: &[String]) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let mut sorted_files: Vec<_> = source_files.iter().collect();
    sorted_files.sort();

    for file in sorted_files {
        let path = Path::new(file);
        if path.exists() {
            let content =
                fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", file, e))?;
            hasher.update(file.as_bytes());
            hasher.update(content.as_bytes());
        }
    }
    Ok(format!("sha256:{}", &hex::encode(hasher.finalize())[..16]))
}

fn compute_global_code_hash() -> Result<String, String> {
    let mut hasher = Sha256::new();
    let src_dir = Path::new("src");

    if src_dir.exists() {
        let mut files: Vec<_> = fs::read_dir(src_dir)
            .map_err(|e| format!("Failed to read src dir: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
            .collect();
        files.sort_by_key(|e| e.path());

        for entry in files {
            let path = entry.path();
            let content = fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            hasher.update(path.to_string_lossy().as_bytes());
            hasher.update(content.as_bytes());
        }
    }

    // Also hash the example itself
    let example_path = Path::new("examples/discover_heuristics.rs");
    if example_path.exists() {
        let content = fs::read_to_string(example_path)
            .map_err(|e| format!("Failed to read example: {}", e))?;
        hasher.update(example_path.to_string_lossy().as_bytes());
        hasher.update(content.as_bytes());
    }

    Ok(format!("sha256:{}", &hex::encode(hasher.finalize())[..16]))
}

fn get_git_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

// ============================================================================
// Cache Management
// ============================================================================

fn load_or_create_manifest(output_dir: &Path) -> Result<CacheManifest, String> {
    let manifest_path = output_dir.join("cache_manifest.json");

    if manifest_path.exists() {
        let content = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read manifest: {}", e))?;
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse manifest: {}", e))
    } else {
        Ok(CacheManifest {
            global_code_hash: compute_global_code_hash()?,
            git_commit: get_git_commit(),
            configs: HashMap::new(),
            last_updated: Utc::now(),
        })
    }
}

fn save_manifest(manifest: &CacheManifest, output_dir: &Path) -> Result<(), String> {
    let manifest_path = output_dir.join("cache_manifest.json");
    let content = serde_json::to_string_pretty(manifest)
        .map_err(|e| format!("Failed to serialize manifest: {}", e))?;
    atomic_write(&manifest_path, content.as_bytes())
}

fn validate_or_update_manifest(
    manifest: &mut CacheManifest,
    configs: &[Config],
    _args: &Args,
) -> Result<(), String> {
    let current_global = compute_global_code_hash()?;
    let git_commit = get_git_commit();

    for config in configs {
        let source_files: Vec<String> = config
            .source_files()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let current_hash = compute_config_code_hash(&source_files)?;
        let key = config.key();
        let version_info = config.version_info();
        let source_version = version_info.version;

        if let Some(entry) = manifest.configs.get(&key) {
            // Check if source version was bumped
            if source_version > entry.version {
                // Version bump in source - update cache entry
                println!(
                    "Config '{}' version bumped: {} -> {} (invalidating cache)",
                    key, entry.version, source_version
                );
                manifest.configs.insert(
                    key.clone(),
                    ConfigCacheEntry {
                        version: source_version,
                        code_hash: current_hash,
                        git_commit: git_commit.clone(),
                        source_files,
                    },
                );
            } else if entry.code_hash != current_hash && !ASSERT_UNCHANGED {
                // Code changed but version not bumped - error with detailed info
                return Err(format!(
                    "Config '{}' code changed but version not bumped!\n\n\
                     Cache state:\n\
                       old_hash    = \"{}\"\n\
                       old_commit  = \"{}\"\n\
                       old_version = {}\n\n\
                     Current state:\n\
                       new_hash    = \"{}\"\n\
                       new_commit  = \"{}\"\n\n\
                     To fix: Update Config::version_info() match arm for this config:\n\
                       VersionInfo {{\n\
                           version: {},\n\
                           old_hash: \"{}\",\n\
                           old_commit: \"{}\",\n\
                       }}\n\n\
                     Or set ASSERT_UNCHANGED = true if changes are non-functional.",
                    key,
                    entry.code_hash,
                    entry.git_commit.as_deref().unwrap_or(""),
                    entry.version,
                    current_hash,
                    git_commit.as_deref().unwrap_or(""),
                    entry.version + 1,
                    entry.code_hash,
                    entry.git_commit.as_deref().unwrap_or(""),
                ));
            }
            // else: hash matches or ASSERT_UNCHANGED - keep existing entry
        } else {
            // New config, add it
            manifest.configs.insert(
                key,
                ConfigCacheEntry {
                    version: source_version,
                    code_hash: current_hash,
                    git_commit: git_commit.clone(),
                    source_files,
                },
            );
        }
    }

    manifest.global_code_hash = current_global;
    manifest.git_commit = git_commit;
    manifest.last_updated = Utc::now();
    Ok(())
}

// ============================================================================
// File Operations
// ============================================================================

fn atomic_write(path: &Path, data: &[u8]) -> Result<(), String> {
    let temp_path = path.with_extension("tmp");

    let mut file =
        File::create(&temp_path).map_err(|e| format!("Failed to create temp file: {}", e))?;
    file.write_all(data)
        .map_err(|e| format!("Failed to write temp file: {}", e))?;
    file.sync_all()
        .map_err(|e| format!("Failed to sync temp file: {}", e))?;
    drop(file);

    fs::rename(&temp_path, path).map_err(|e| format!("Failed to rename temp file: {}", e))?;
    Ok(())
}

fn compute_source_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())[..12].to_string()
}

fn format_encoding_filename(
    bpp: f32,
    ssim2: f32,
    ba: f32,
    config_key: &str,
    quality: u8,
    version: u32,
) -> String {
    format!(
        "{:.3}bpp_{:.1}ss_{:.2}ba_{}-q{}_v{}.jpg",
        bpp, ssim2, ba, config_key, quality, version
    )
}

fn format_metrics_filename(
    bpp: f32,
    ssim2: f32,
    ba: f32,
    config_key: &str,
    quality: u8,
    version: u32,
) -> String {
    format!(
        "{:.3}bpp_{:.1}ss_{:.2}ba_{}-q{}_v{}.json",
        bpp, ssim2, ba, config_key, quality, version
    )
}

// ============================================================================
// Image Analysis
// ============================================================================

fn analyze_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    source_hash: &str,
    source_name: &str,
) -> ImageAnalysis {
    // Calculate luminance and stats
    let luma: Vec<f32> = pixels
        .chunks(3)
        .map(|rgb| 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
        .collect();

    let mean: f32 = luma.iter().sum::<f32>() / luma.len() as f32;
    let variance = luma.iter().map(|&l| (l - mean).powi(2)).sum::<f32>() / luma.len() as f32;

    // Edge density
    let mut edge_sum = 0.0f32;
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = y * width + x;
            if idx + width < luma.len() && idx > 0 {
                let gx = (luma[idx + 1] - luma[idx - 1]).abs();
                let gy = (luma[idx + width]
                    - luma.get(idx.saturating_sub(width)).copied().unwrap_or(0.0))
                .abs();
                edge_sum += (gx * gx + gy * gy).sqrt();
            }
        }
    }
    let edge_density =
        edge_sum / ((width.saturating_sub(2)) * (height.saturating_sub(2))) as f32 / 255.0;

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

    // High frequency detection
    let has_high_frequency = edge_density > 0.15;

    // Color count estimate (simplified)
    let color_count_estimate = (pixels.len() / 3).min(100000) as u32;

    ImageAnalysis {
        source_hash: source_hash.to_string(),
        source_name: source_name.to_string(),
        width,
        height,
        pixels: width * height,
        variance,
        edge_density,
        chroma_complexity,
        uniform_block_fraction,
        has_high_frequency,
        color_count_estimate,
        timestamp: Utc::now(),
    }
}

// ============================================================================
// Encoding
// ============================================================================

fn encode_mozjpeg(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    subsampling: Subsampling,
    max_compression: bool,
) -> Result<Vec<u8>, String> {
    use mozjpeg_oxide::Encoder;

    let subsamp = match subsampling {
        Subsampling::S420 | Subsampling::Auto => mozjpeg_oxide::Subsampling::S420,
        Subsampling::S422 => mozjpeg_oxide::Subsampling::S422,
        Subsampling::S444 => mozjpeg_oxide::Subsampling::S444,
    };

    let encoder = if max_compression {
        // Progressive + optimize_scans for maximum compression
        Encoder::max_compression()
            .quality(quality)
            .subsampling(subsamp)
    } else {
        // Baseline optimized mode
        Encoder::baseline_optimized()
            .quality(quality)
            .subsampling(subsamp)
    };

    encoder
        .encode_rgb(pixels, width as u32, height as u32)
        .map_err(|e| format!("mozjpeg encode failed: {:?}", e))
}

/// Encode using C mozjpeg (reference implementation via mozjpeg crate)
///
/// C mozjpeg defaults (via jpeg_set_defaults):
/// - Trellis quantization: ENABLED by default
/// - Trellis DC: ENABLED by default
/// - Trellis EOB opt: ENABLED by default
/// - Overshoot deringing: ENABLED by default
///
/// We additionally set:
/// - optimize_coding: true (Huffman optimization)
/// - progressive + optimize_scans (for max_compression mode)
fn encode_cmozjpeg(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    subsampling: Subsampling,
    max_compression: bool,
) -> Result<Vec<u8>, String> {
    use mozjpeg::{ColorSpace, Compress};

    let mut comp = Compress::new(ColorSpace::JCS_RGB);
    comp.set_size(width, height);
    comp.set_quality(quality as f32);

    // Enable Huffman optimization (like mozjpeg-rs does)
    comp.set_optimize_coding(true);

    // Set subsampling using pixel sizes: (h, v) where (2,2) = 4:2:0, (2,1) = 4:2:2, (1,1) = 4:4:4
    match subsampling {
        Subsampling::S420 | Subsampling::Auto => {
            comp.set_chroma_sampling_pixel_sizes((2, 2), (2, 2));
        }
        Subsampling::S422 => {
            comp.set_chroma_sampling_pixel_sizes((2, 1), (2, 1));
        }
        Subsampling::S444 => {
            comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));
        }
    }

    if max_compression {
        // Enable progressive and scan optimization for max compression
        comp.set_progressive_mode();
        comp.set_optimize_scans(true);
    }

    // Start compression
    let mut comp = comp
        .start_compress(Vec::new())
        .map_err(|e| format!("cmozjpeg start failed: {:?}", e))?;

    // Write all scanlines at once (the API handles chunking internally)
    comp.write_scanlines(pixels)
        .map_err(|e| format!("cmozjpeg scanlines failed: {:?}", e))?;

    comp.finish()
        .map_err(|e| format!("cmozjpeg finish failed: {:?}", e))
}

fn encode_jpegli(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    subsampling: Subsampling,
    _xyb_mode: bool,
) -> Result<Vec<u8>, String> {
    use jpegli::{Encoder, Quality, Subsampling as JpegliSubsampling};

    let subsamp = match subsampling {
        Subsampling::S420 | Subsampling::Auto => JpegliSubsampling::S420,
        Subsampling::S422 => JpegliSubsampling::S422,
        Subsampling::S444 => JpegliSubsampling::S444,
    };

    let encoder = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(Quality::from_quality(quality as f32))
        .subsampling(subsamp);

    encoder
        .encode(pixels)
        .map_err(|e| format!("jpegli encode failed: {:?}", e))
}

fn encode_zenjpeg(
    _pixels: &[u8],
    _width: usize,
    _height: usize,
    _quality: u8,
    _subsampling: Subsampling,
) -> Result<Vec<u8>, String> {
    // TODO: Implement zenjpeg encoding when zenjpeg library is ready
    Err("zenjpeg encoder not yet implemented".to_string())
}

fn decode_jpeg(data: &[u8]) -> Result<Vec<u8>, String> {
    use jpeg_decoder::Decoder;

    let mut decoder = Decoder::new(std::io::Cursor::new(data));
    decoder
        .decode()
        .map_err(|e| format!("JPEG decode failed: {:?}", e))
}

/// Convert RGB8 slice to Vec<[u8; 3]> for fast-ssim2
fn rgb8_to_array(data: &[u8]) -> Vec<[u8; 3]> {
    data.chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

/// Metric results with timing breakdown
struct MetricResults {
    butteraugli: f32,
    ssimulacra2: f32,
    dssim: f32,
    butteraugli_ms: u64,
    ssim2_ms: u64,
    dssim_ms: u64,
}

// ============================================================================
// ImageProcessor - Lockstep processing with cached metric references
// ============================================================================

use butteraugli::{ButteraugliParams, precompute::ButteraugliReference};
use codec_eval::metrics::dssim::{calculate_dssim, rgb8_to_dssim_image};
use codec_eval::viewing::ViewingCondition;
use fast_ssim2::Ssimulacra2Reference;
use imgref::Img;

// GPU-accelerated SSIM2 support (requires --features gpu)
#[cfg(feature = "gpu")]
use cudarse_driver::CuStream;
#[cfg(feature = "gpu")]
use cudarse_npp::image::isu::Malloc;
#[cfg(feature = "gpu")]
use cudarse_npp::image::{Image as NppImage, Img as NppImg, ImgMut, C};
#[cfg(feature = "gpu")]
use cudarse_npp::set_stream;
#[cfg(feature = "gpu")]
use ssimulacra2_cuda::Ssimulacra2 as GpuSsimulacra2;
#[cfg(feature = "gpu")]
use dssim_cuda::Dssim as GpuDssim;
#[cfg(feature = "gpu")]
use butteraugli_cuda::Butteraugli as GpuButteraugli;

/// GPU verification constants - based on observed max errors from 24-image Kodak benchmark
#[cfg(feature = "gpu")]
const GPU_VERIFY_INTERVAL: usize = 50;  // Verify every N tests
#[cfg(feature = "gpu")]
const GPU_SSIM2_EPSILON_PCT: f64 = 0.5;  // Max observed: 0.43% (relative error higher near zero values)
#[cfg(feature = "gpu")]
const GPU_SSIM2_EPSILON_ABS: f64 = 0.1;  // Absolute threshold for near-zero SSIM2 scores
#[cfg(feature = "gpu")]
const GPU_DSSIM_EPSILON_PCT: f64 = 0.1;  // Max observed: 0.045%
#[cfg(feature = "gpu")]
const GPU_BUTTERAUGLI_EPSILON_PCT: f64 = 12.0;  // Max observed: 11.37% (multi-scale algorithm variance)

/// Initialize CUDA once at startup
#[cfg(feature = "gpu")]
fn init_cuda_once() -> bool {
    static INIT: std::sync::Once = std::sync::Once::new();
    static mut SUCCESS: bool = false;

    INIT.call_once(|| {
        unsafe {
            SUCCESS = cudarse_driver::init_cuda_and_primary_ctx().is_ok();
        }
    });

    unsafe { SUCCESS }
}

/// GPU-accelerated SSIM2 context
#[cfg(feature = "gpu")]
struct GpuSsim2Context {
    stream: CuStream,
    tmp_ref: NppImage<u8, C<3>>,
    tmp_dis: NppImage<u8, C<3>>,
    ref_linear: NppImage<f32, C<3>>,
    dis_linear: NppImage<f32, C<3>>,
    ssimulacra2: GpuSsimulacra2,
}

#[cfg(feature = "gpu")]
impl GpuSsim2Context {
    fn new(width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = CuStream::new()?;
        set_stream(stream.inner() as _)?;

        // Allocate GPU buffers
        let tmp_ref: NppImage<u8, C<3>> = NppImage::malloc(width, height)?;
        let tmp_dis: NppImage<u8, C<3>> = tmp_ref.malloc_same_size()?;
        let ref_linear: NppImage<f32, C<3>> = NppImage::malloc(width, height)?;
        let dis_linear: NppImage<f32, C<3>> = ref_linear.malloc_same_size()?;

        // Create ssimulacra2 instance (tied to these dimensions)
        let ssimulacra2 = GpuSsimulacra2::new(&ref_linear, &dis_linear, &stream)?;

        // Sync to ensure all GPU operations complete before first compute
        stream.sync()?;

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
        // Verify buffer sizes match
        let expected = self.tmp_ref.width() as usize * self.tmp_ref.height() as usize * 3;
        if reference.len() != expected || distorted.len() != expected {
            eprintln!("GPU SSIM2: size mismatch: ref={}, dis={}, expected={}",
                reference.len(), distorted.len(), expected);
            return 0.0;
        }

        // compute_from_cpu_srgb_sync handles upload and sRGB->linear conversion
        match self.ssimulacra2.compute_from_cpu_srgb_sync(
            reference,
            distorted,
            &mut self.tmp_ref,
            &mut self.tmp_dis,
            &mut self.ref_linear,
            &mut self.dis_linear,
            &self.stream
        ) {
            Ok(score) => score,
            Err(e) => {
                eprintln!("GPU SSIM2 compute error: {:?}", e);
                0.0
            }
        }
    }

    /// Sync and cleanup before dropping
    fn cleanup(&self) {
        // Sync to ensure all GPU operations complete before dropping resources
        let _ = self.stream.sync();
        // Set default stream to avoid NPP using our stream during cleanup
        let _ = set_stream(CuStream::DEFAULT.inner() as _);
    }
}

#[cfg(feature = "gpu")]
impl Drop for GpuSsim2Context {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// GPU-accelerated DSSIM context
#[cfg(feature = "gpu")]
struct GpuDssimContext {
    stream: CuStream,
    tmp_ref: NppImage<u8, C<3>>,
    tmp_dis: NppImage<u8, C<3>>,
    dssim: GpuDssim,
}

#[cfg(feature = "gpu")]
impl GpuDssimContext {
    fn new(width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = CuStream::new()?;
        set_stream(stream.inner() as _)?;

        // Allocate GPU buffers for sRGB images
        let tmp_ref: NppImage<u8, C<3>> = NppImage::malloc(width, height)?;
        let tmp_dis: NppImage<u8, C<3>> = tmp_ref.malloc_same_size()?;

        // Create DSSIM instance (tied to these dimensions)
        let dssim = GpuDssim::new(width, height, &stream)?;

        // Sync to ensure all GPU operations complete before first compute
        stream.sync()?;

        Ok(Self {
            stream,
            tmp_ref,
            tmp_dis,
            dssim,
        })
    }

    fn compute(&mut self, reference: &[u8], distorted: &[u8]) -> f64 {
        // Verify buffer sizes match
        let expected = self.tmp_ref.width() as usize * self.tmp_ref.height() as usize * 3;
        if reference.len() != expected || distorted.len() != expected {
            eprintln!("GPU DSSIM: size mismatch: ref={}, dis={}, expected={}",
                reference.len(), distorted.len(), expected);
            return 0.0;
        }

        // Upload images to GPU
        if let Err(e) = self.tmp_ref.copy_from_cpu(reference, self.stream.inner() as _) {
            eprintln!("GPU DSSIM: failed to upload reference: {:?}", e);
            return 0.0;
        }
        if let Err(e) = self.tmp_dis.copy_from_cpu(distorted, self.stream.inner() as _) {
            eprintln!("GPU DSSIM: failed to upload distorted: {:?}", e);
            return 0.0;
        }

        // Compute DSSIM
        match self.dssim.compute_sync(&self.tmp_ref, &self.tmp_dis, &self.stream) {
            Ok(score) => score,
            Err(e) => {
                eprintln!("GPU DSSIM compute error: {:?}", e);
                0.0
            }
        }
    }

    /// Sync and cleanup before dropping
    fn cleanup(&self) {
        let _ = self.stream.sync();
        let _ = set_stream(CuStream::DEFAULT.inner() as _);
    }
}

#[cfg(feature = "gpu")]
impl Drop for GpuDssimContext {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// GPU-accelerated Butteraugli context
#[cfg(feature = "gpu")]
struct GpuButteraugliContext {
    tmp_ref: NppImage<u8, C<3>>,
    tmp_dis: NppImage<u8, C<3>>,
    butteraugli: GpuButteraugli,
}

#[cfg(feature = "gpu")]
impl GpuButteraugliContext {
    fn new(width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        // Allocate GPU buffers for sRGB images
        let tmp_ref: NppImage<u8, C<3>> = NppImage::malloc(width, height)?;
        let tmp_dis: NppImage<u8, C<3>> = tmp_ref.malloc_same_size()?;

        // Create Butteraugli instance (has its own internal stream)
        let butteraugli = GpuButteraugli::new(width, height)?;

        Ok(Self {
            tmp_ref,
            tmp_dis,
            butteraugli,
        })
    }

    fn compute(&mut self, reference: &[u8], distorted: &[u8], stream: &CuStream) -> f32 {
        // Verify buffer sizes match
        let expected = self.tmp_ref.width() as usize * self.tmp_ref.height() as usize * 3;
        if reference.len() != expected || distorted.len() != expected {
            eprintln!("GPU Butteraugli: size mismatch: ref={}, dis={}, expected={}",
                reference.len(), distorted.len(), expected);
            return f32::MAX;
        }

        // Upload images to GPU
        if let Err(e) = self.tmp_ref.copy_from_cpu(reference, stream.inner() as _) {
            eprintln!("GPU Butteraugli: failed to upload reference: {:?}", e);
            return f32::MAX;
        }
        if let Err(e) = self.tmp_dis.copy_from_cpu(distorted, stream.inner() as _) {
            eprintln!("GPU Butteraugli: failed to upload distorted: {:?}", e);
            return f32::MAX;
        }

        // CRITICAL: Sync before compute - Butteraugli has its own internal stream,
        // so we must ensure uploads are complete before it starts computing
        if let Err(e) = stream.sync() {
            eprintln!("GPU Butteraugli: failed to sync stream: {:?}", e);
            return f32::MAX;
        }

        // Compute Butteraugli
        match self.butteraugli.compute(self.tmp_ref.full_view(), self.tmp_dis.full_view()) {
            Ok(score) => score,
            Err(e) => {
                eprintln!("GPU Butteraugli compute error: {:?}", e);
                f32::MAX
            }
        }
    }

    /// Sync and cleanup before dropping
    fn cleanup(&self, stream: &CuStream) {
        let _ = stream.sync();
        let _ = set_stream(CuStream::DEFAULT.inner() as _);
    }
}

/// Holds cached metric references for efficient repeated comparisons.
///
/// When processing a single source image through multiple codecs/qualities,
/// the reference image data only needs to be processed once. This provides
/// ~40-50% speedup for butteraugli and significant speedup for SSIM2.
///
/// When GPU mode is enabled, SSIM2 and DSSIM use GPU acceleration for faster computation.
struct ImageProcessor {
    /// Original RGB pixels (sRGB u8) - owned copy for GPU metrics
    rgb_pixels: Vec<u8>,
    /// RGB as array for CPU SSIM2 - owned
    rgb_array: Vec<[u8; 3]>,
    /// Image dimensions
    width: usize,
    height: usize,
    /// Cached butteraugli reference (precomputed XYB + frequency decomposition)
    butteraugli_ref: ButteraugliReference,
    /// Cached SSIM2 reference (precomputed linear RGB) - owns its data (CPU mode)
    ssim2_ref: Ssimulacra2Reference,
    /// Cached DSSIM reference image (ImgVec) - for CPU mode
    dssim_ref_img: imgref::ImgVec<rgb::RGBA<f32>>,
    /// Viewing condition for DSSIM
    viewing: ViewingCondition,
    /// GPU SSIM2 context (when --gpu flag is used)
    /// Using Mutex for Send+Sync, required for rayon parallel iteration
    #[cfg(feature = "gpu")]
    gpu_ssim2: Option<Mutex<GpuSsim2Context>>,
    /// GPU DSSIM context (when --gpu flag is used)
    #[cfg(feature = "gpu")]
    gpu_dssim: Option<Mutex<GpuDssimContext>>,
    /// GPU Butteraugli context (when --gpu flag is used)
    #[cfg(feature = "gpu")]
    gpu_butteraugli: Option<Mutex<GpuButteraugliContext>>,
    /// Shared CUDA stream for GPU operations
    #[cfg(feature = "gpu")]
    gpu_stream: Option<CuStream>,
    /// Counter for GPU verification (verify every N tests)
    #[cfg(feature = "gpu")]
    verification_counter: AtomicUsize,
}

impl ImageProcessor {
    /// Create a new ImageProcessor with cached metric references.
    ///
    /// This precomputes reference data for butteraugli and SSIM2, which is expensive
    /// but pays off when comparing against many distorted versions.
    ///
    /// If `use_gpu` is true and the GPU feature is enabled, SSIM2 will use GPU acceleration.
    fn new(rgb_pixels: Vec<u8>, width: usize, height: usize, use_gpu: bool) -> Result<Self, String> {
        // Convert to array format for SSIM2
        let rgb_array: Vec<[u8; 3]> = rgb_pixels
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        // Create butteraugli reference (precomputes XYB + frequency decomposition)
        // ButteraugliReference::new takes &[u8] and copies internally
        let butteraugli_ref = ButteraugliReference::new(
            &rgb_pixels,
            width,
            height,
            ButteraugliParams::default(),
        ).map_err(|e| format!("Failed to create butteraugli reference: {}", e))?;

        // Create SSIM2 reference (precomputes linear RGB conversion)
        // Ssimulacra2Reference::new takes Img<&[[u8;3]]> and copies internally
        let ref_img = Img::new(rgb_array.as_slice(), width, height);
        let ssim2_ref = Ssimulacra2Reference::new(ref_img)
            .map_err(|e| format!("Failed to create SSIM2 reference: {:?}", e))?;

        // Create DSSIM reference image (just the ImgVec, comparator is stateless)
        let dssim_ref_img = rgb8_to_dssim_image(&rgb_pixels, width, height);
        let viewing = ViewingCondition::default();

        // Initialize GPU contexts if requested
        #[cfg(feature = "gpu")]
        let gpu_ssim2 = if use_gpu {
            eprintln!("  Creating GPU SSIM2 context for {}x{}...", width, height);
            match GpuSsim2Context::new(width as u32, height as u32) {
                Ok(ctx) => {
                    eprintln!("  GPU SSIM2 context created successfully");
                    Some(Mutex::new(ctx))
                }
                Err(e) => {
                    eprintln!("Warning: Failed to create GPU SSIM2 context: {}. Falling back to CPU.", e);
                    None
                }
            }
        } else {
            None
        };

        #[cfg(feature = "gpu")]
        let gpu_dssim = if use_gpu {
            eprintln!("  Creating GPU DSSIM context for {}x{}...", width, height);
            match GpuDssimContext::new(width as u32, height as u32) {
                Ok(ctx) => {
                    eprintln!("  GPU DSSIM context created successfully");
                    Some(Mutex::new(ctx))
                }
                Err(e) => {
                    eprintln!("Warning: Failed to create GPU DSSIM context: {}. Falling back to CPU.", e);
                    None
                }
            }
        } else {
            None
        };

        #[cfg(feature = "gpu")]
        let gpu_butteraugli = if use_gpu {
            eprintln!("  Creating GPU Butteraugli context for {}x{}...", width, height);
            match GpuButteraugliContext::new(width as u32, height as u32) {
                Ok(ctx) => {
                    eprintln!("  GPU Butteraugli context created successfully");
                    Some(Mutex::new(ctx))
                }
                Err(e) => {
                    eprintln!("Warning: Failed to create GPU Butteraugli context: {}. Falling back to CPU.", e);
                    None
                }
            }
        } else {
            None
        };

        #[cfg(feature = "gpu")]
        let gpu_stream = if use_gpu {
            CuStream::new().ok()
        } else {
            None
        };

        #[cfg(not(feature = "gpu"))]
        if use_gpu {
            eprintln!("Warning: --gpu requested but GPU feature not enabled. Using CPU.");
        }

        Ok(Self {
            rgb_pixels,
            rgb_array,
            width,
            height,
            butteraugli_ref,
            ssim2_ref,
            dssim_ref_img,
            viewing,
            #[cfg(feature = "gpu")]
            gpu_ssim2,
            #[cfg(feature = "gpu")]
            gpu_dssim,
            #[cfg(feature = "gpu")]
            gpu_butteraugli,
            #[cfg(feature = "gpu")]
            gpu_stream,
            #[cfg(feature = "gpu")]
            verification_counter: AtomicUsize::new(0),
        })
    }

    /// Measure all metrics against the cached reference.
    ///
    /// Butteraugli uses cached reference for ~40-50% speedup.
    /// When GPU is enabled, SSIM2 and DSSIM use GPU acceleration.
    fn measure(&self, decoded: &[u8]) -> MetricResults {
        let expected_size = self.width * self.height * 3;

        if decoded.len() < expected_size {
            return MetricResults {
                butteraugli: f32::MAX,
                ssimulacra2: 0.0,
                dssim: f32::MAX,
                butteraugli_ms: 0,
                ssim2_ms: 0,
                dssim_ms: 0,
            };
        }

        let decoded_slice = &decoded[..expected_size];

        // Check GPU availability and if verification is needed
        #[cfg(feature = "gpu")]
        let use_gpu_ssim2 = self.gpu_ssim2.is_some();
        #[cfg(not(feature = "gpu"))]
        let use_gpu_ssim2 = false;

        #[cfg(feature = "gpu")]
        let use_gpu_dssim = self.gpu_dssim.is_some();
        #[cfg(not(feature = "gpu"))]
        let use_gpu_dssim = false;

        #[cfg(feature = "gpu")]
        let use_gpu_butteraugli = self.gpu_butteraugli.is_some();
        #[cfg(not(feature = "gpu"))]
        let use_gpu_butteraugli = false;

        #[cfg(feature = "gpu")]
        let should_verify = {
            let count = self.verification_counter.fetch_add(1, Ordering::Relaxed);
            (use_gpu_ssim2 || use_gpu_dssim || use_gpu_butteraugli) && (count % GPU_VERIFY_INTERVAL == 0)
        };

        // Calculate SSIM2 - either on GPU or CPU
        let (ssimulacra2, ssim_ms) = if use_gpu_ssim2 {
            #[cfg(feature = "gpu")]
            {
                let start = Instant::now();
                let result = self.gpu_ssim2.as_ref().unwrap().lock().unwrap()
                    .compute(&self.rgb_pixels, decoded_slice) as f32;
                (result, start.elapsed().as_millis() as u64)
            }
            #[cfg(not(feature = "gpu"))]
            unreachable!()
        } else {
            let start = Instant::now();
            let test_arr = rgb8_to_array(decoded_slice);
            let test_img = Img::new(test_arr.as_slice(), self.width, self.height);
            let result = self.ssim2_ref.compare(test_img)
                .map(|s| s as f32)
                .unwrap_or(0.0);
            (result, start.elapsed().as_millis() as u64)
        };

        // Calculate DSSIM - either on GPU or CPU
        let (dssim, dssim_ms) = if use_gpu_dssim {
            #[cfg(feature = "gpu")]
            {
                let start = Instant::now();
                let result = self.gpu_dssim.as_ref().unwrap().lock().unwrap()
                    .compute(&self.rgb_pixels, decoded_slice) as f32;
                (result, start.elapsed().as_millis() as u64)
            }
            #[cfg(not(feature = "gpu"))]
            unreachable!()
        } else {
            let start = Instant::now();
            let test_img = rgb8_to_dssim_image(decoded_slice, self.width, self.height);
            let result = calculate_dssim(&self.dssim_ref_img, &test_img, &self.viewing)
                .map(|s| s as f32)
                .unwrap_or(f32::MAX);
            (result, start.elapsed().as_millis() as u64)
        };

        // Calculate Butteraugli - either on GPU or CPU
        let (butteraugli, ba_ms) = if use_gpu_butteraugli {
            #[cfg(feature = "gpu")]
            {
                let start = Instant::now();
                let stream = self.gpu_stream.as_ref().expect("GPU stream required for butteraugli");
                let result = self.gpu_butteraugli.as_ref().unwrap().lock().unwrap()
                    .compute(&self.rgb_pixels, decoded_slice, stream);
                (result, start.elapsed().as_millis() as u64)
            }
            #[cfg(not(feature = "gpu"))]
            unreachable!()
        } else {
            let start = Instant::now();
            let result = self.butteraugli_ref.compare(decoded_slice)
                .map(|r| r.score as f32)
                .unwrap_or(f32::MAX);
            (result, start.elapsed().as_millis() as u64)
        };

        // GPU verification: compare GPU results against CPU every N tests
        #[cfg(feature = "gpu")]
        if should_verify {
            self.verify_gpu_accuracy(decoded_slice, ssimulacra2, dssim, butteraugli,
                                     use_gpu_ssim2, use_gpu_dssim, use_gpu_butteraugli);
        }

        MetricResults {
            butteraugli,
            ssimulacra2,
            dssim,
            butteraugli_ms: ba_ms,
            ssim2_ms: ssim_ms,
            dssim_ms,
        }
    }

    /// Verify GPU results against CPU computation
    #[cfg(feature = "gpu")]
    fn verify_gpu_accuracy(&self, decoded_slice: &[u8], gpu_ssim2: f32, gpu_dssim: f32, gpu_butteraugli: f32,
                           used_gpu_ssim2: bool, used_gpu_dssim: bool, used_gpu_butteraugli: bool) {
        let count = self.verification_counter.load(Ordering::Relaxed);

        // Verify SSIM2
        if used_gpu_ssim2 {
            let test_arr = rgb8_to_array(decoded_slice);
            let test_img = Img::new(test_arr.as_slice(), self.width, self.height);
            let cpu_ssim2 = self.ssim2_ref.compare(test_img)
                .map(|s| s as f32)
                .unwrap_or(0.0);

            let abs_diff = (gpu_ssim2 - cpu_ssim2).abs();
            let error_pct = if cpu_ssim2.abs() > 0.0001 {
                (abs_diff / cpu_ssim2.abs()) * 100.0
            } else {
                abs_diff * 100.0
            };

            // Use absolute threshold for small values where relative error is misleading
            // SSIM2 scores range from ~-inf to 100, with ~70+ being "good quality"
            // For low scores (< 10), absolute diff matters more than percentage
            let is_valid = if cpu_ssim2.abs() < 10.0 {
                abs_diff < GPU_SSIM2_EPSILON_ABS as f32
            } else {
                error_pct <= GPU_SSIM2_EPSILON_PCT as f32
            };

            if !is_valid {
                eprintln!("\n🚨🚨🚨 GPU SSIM2 DIVERGENCE DETECTED! 🚨🚨🚨");
                eprintln!("   Test #{}: GPU={:.6} CPU={:.6} Error={:.3}% abs={:.4} (max={:.1}%/{:.2}abs)",
                         count, gpu_ssim2, cpu_ssim2, error_pct, abs_diff, GPU_SSIM2_EPSILON_PCT, GPU_SSIM2_EPSILON_ABS);
                eprintln!("   ⚠️  Results may be unreliable! Consider using --no-gpu\n");
            } else {
                eprintln!("✅ GPU SSIM2 verified #{}: GPU={:.4} CPU={:.4} err={:.4}%",
                         count, gpu_ssim2, cpu_ssim2, error_pct);
            }
        }

        // Verify DSSIM
        if used_gpu_dssim {
            let test_img = rgb8_to_dssim_image(decoded_slice, self.width, self.height);
            let cpu_dssim = calculate_dssim(&self.dssim_ref_img, &test_img, &self.viewing)
                .map(|s| s as f32)
                .unwrap_or(f32::MAX);

            let error_pct = if cpu_dssim.abs() > 0.0001 {
                ((gpu_dssim - cpu_dssim).abs() / cpu_dssim.abs()) * 100.0
            } else {
                (gpu_dssim - cpu_dssim).abs() * 100.0
            };

            if error_pct > GPU_DSSIM_EPSILON_PCT as f32 {
                eprintln!("\n🚨🚨🚨 GPU DSSIM DIVERGENCE DETECTED! 🚨🚨🚨");
                eprintln!("   Test #{}: GPU={:.6} CPU={:.6} Error={:.3}% (max={:.1}%)",
                         count, gpu_dssim, cpu_dssim, error_pct, GPU_DSSIM_EPSILON_PCT);
                eprintln!("   ⚠️  Results may be unreliable! Consider using --no-gpu\n");
            } else {
                eprintln!("✅ GPU DSSIM verified #{}: GPU={:.6} CPU={:.6} err={:.4}%",
                         count, gpu_dssim, cpu_dssim, error_pct);
            }
        }

        // Verify Butteraugli
        if used_gpu_butteraugli {
            let cpu_butteraugli = self.butteraugli_ref.compare(decoded_slice)
                .map(|r| r.score as f32)
                .unwrap_or(f32::MAX);

            let error_pct = if cpu_butteraugli.abs() > 0.0001 {
                ((gpu_butteraugli - cpu_butteraugli).abs() / cpu_butteraugli.abs()) * 100.0
            } else {
                (gpu_butteraugli - cpu_butteraugli).abs() * 100.0
            };

            if error_pct > GPU_BUTTERAUGLI_EPSILON_PCT as f32 {
                eprintln!("\n🚨🚨🚨 GPU BUTTERAUGLI DIVERGENCE DETECTED! 🚨🚨🚨");
                eprintln!("   Test #{}: GPU={:.4} CPU={:.4} Error={:.3}% (max={:.1}%)",
                         count, gpu_butteraugli, cpu_butteraugli, error_pct, GPU_BUTTERAUGLI_EPSILON_PCT);
                eprintln!("   ⚠️  Results may be unreliable! Consider using --no-gpu\n");
            } else {
                eprintln!("✅ GPU Butteraugli verified #{}: GPU={:.4} CPU={:.4} err={:.2}%",
                         count, gpu_butteraugli, cpu_butteraugli, error_pct);
            }
        }
    }
}

/// Process a single source image through all configs and qualities (lockstep mode).
///
/// This is the lockstep processing model: one image at a time, with cached
/// metric references shared across all encodings. This provides ~40-50%
/// speedup for metric calculation by caching butteraugli and SSIM2 reference data.
///
/// Returns a vector of WorkResults for all processed items.
fn process_image_lockstep(
    work_items: &[WorkItem],
    stats: &AtomicRunStats,
    args: &Args,
) -> Vec<WorkResult> {
    if work_items.is_empty() {
        return Vec::new();
    }

    // All work items share the same source image
    let first = &work_items[0];
    let rgb_pixels = first.rgb_pixels.as_ref().clone();
    let width = first.width;
    let height = first.height;
    let analysis = Arc::clone(&first.analysis);

    // Create the ImageProcessor with cached metric references (and GPU if enabled)
    let processor = match ImageProcessor::new(rgb_pixels, width, height, args.gpu) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create ImageProcessor: {}", e);
            return work_items
                .iter()
                .map(|item| WorkResult {
                    analysis: Arc::clone(&item.analysis),
                    metrics: None,
                    cached: false,
                    error: Some(e.clone()),
                })
                .collect();
        }
    };

    // Process all work items, using the cached processor for metrics
    work_items
        .iter()
        .map(|item| process_work_item_with_processor(item, &processor, &analysis, stats, args))
        .collect()
}

/// Process a single work item using a pre-created ImageProcessor.
///
/// This is the inner loop of lockstep processing - encodes at one quality,
/// decodes, and measures metrics using the cached references.
fn process_work_item_with_processor(
    item: &WorkItem,
    processor: &ImageProcessor,
    analysis: &Arc<ImageAnalysis>,
    stats: &AtomicRunStats,
    args: &Args,
) -> WorkResult {
    let config_key = item.config.key();

    // Check for existing cache
    let cache_filename = format!(
        "{}-q{}_v{}.jpg",
        config_key, item.quality, item.cache_version
    );
    let cached = fs::read_dir(&item.image_dir).ok().and_then(|entries| {
        entries.filter_map(|e| e.ok()).find(|e| {
            e.file_name().to_str().map(|s| s == cache_filename).unwrap_or(false)
        })
    });

    if cached.is_some() && !args.force {
        stats.encodings_cached.fetch_add(1, Ordering::Relaxed);
        return WorkResult {
            analysis: Arc::clone(analysis),
            metrics: None,
            error: None,
            cached: true,
        };
    }

    // Encode
    let encode_start = Instant::now();
    let encode_result = match item.config {
        Config::MozJpeg { subsampling } | Config::MozJpegMax { subsampling } => {
            encode_mozjpeg(
                &processor.rgb_pixels,
                processor.width,
                processor.height,
                item.quality,
                subsampling,
                matches!(item.config, Config::MozJpegMax { .. }),
            )
        }
        Config::CMozJpeg { subsampling } | Config::CMozJpegMax { subsampling } => encode_cmozjpeg(
            &processor.rgb_pixels,
            processor.width,
            processor.height,
            item.quality,
            subsampling,
            matches!(item.config, Config::CMozJpegMax { .. }),
        ),
        Config::Jpegli { subsampling } | Config::JpegliXyb { subsampling } => encode_jpegli(
            &processor.rgb_pixels,
            processor.width,
            processor.height,
            item.quality,
            subsampling,
            matches!(item.config, Config::JpegliXyb { .. }),
        ),
        Config::Zenjpeg { .. } => {
            // Zenjpeg not implemented in lockstep mode yet
            return WorkResult {
                analysis: Arc::clone(analysis),
                metrics: None,
                error: Some("Zenjpeg not implemented in lockstep mode".to_string()),
                cached: false,
            };
        }
    };

    let jpeg_data = match encode_result {
        Ok(data) => data,
        Err(e) => {
            stats
                .errors
                .lock()
                .unwrap()
                .push(format!("{:?} q{}: {}", item.config, item.quality, e));
            return WorkResult {
                analysis: Arc::clone(analysis),
                metrics: None,
                error: Some(e),
                cached: false,
            };
        }
    };
    let encode_ms = encode_start.elapsed().as_millis() as u64;
    stats.encodings_performed.fetch_add(1, Ordering::Relaxed);
    stats.total_encode_time_ms.fetch_add(encode_ms, Ordering::Relaxed);

    // Decode
    let decode_start = Instant::now();
    let decoded = match decode_jpeg(&jpeg_data) {
        Ok(d) => d,
        Err(e) => {
            stats
                .errors
                .lock()
                .unwrap()
                .push(format!("{:?} q{} decode: {}", item.config, item.quality, e));
            return WorkResult {
                analysis: Arc::clone(analysis),
                metrics: None,
                error: Some(e),
                cached: false,
            };
        }
    };
    let decode_ms = decode_start.elapsed().as_millis() as u64;
    stats.total_decode_ms.fetch_add(decode_ms, Ordering::Relaxed);

    // Measure metrics using the cached processor
    let metric_results = processor.measure(&decoded);

    // Accumulate per-metric timing
    stats.total_butteraugli_ms.fetch_add(metric_results.butteraugli_ms, Ordering::Relaxed);
    stats.total_ssim2_ms.fetch_add(metric_results.ssim2_ms, Ordering::Relaxed);
    stats.total_dssim_ms.fetch_add(metric_results.dssim_ms, Ordering::Relaxed);
    stats.total_metric_time_ms.fetch_add(
        metric_results.butteraugli_ms + metric_results.ssim2_ms + metric_results.dssim_ms,
        Ordering::Relaxed,
    );

    let size_bytes = jpeg_data.len();
    let bpp = (size_bytes as f32 * 8.0) / (processor.width * processor.height) as f32;
    let butteraugli = metric_results.butteraugli;
    let ssimulacra2 = metric_results.ssimulacra2;
    let dssim = metric_results.dssim;

    // Create metrics struct
    let metrics = EncodingMetrics {
        source_hash: analysis.source_hash.clone(),
        config_key: config_key.to_string(),
        quality: item.quality,
        cache_version: item.cache_version,
        size_bytes,
        bpp,
        butteraugli,
        ssimulacra2,
        dssim,
        encode_time_ms: encode_ms,
        timestamp: Utc::now(),
    };

    // Write files with metric-based names (matches process_work_item behavior)
    let jpg_name = format_encoding_filename(
        bpp,
        ssimulacra2,
        butteraugli,
        &config_key,
        item.quality,
        item.cache_version,
    );
    let json_name = format_metrics_filename(
        bpp,
        ssimulacra2,
        butteraugli,
        &config_key,
        item.quality,
        item.cache_version,
    );

    if let Err(e) = atomic_write(&item.image_dir.join(&jpg_name), &jpeg_data) {
        return WorkResult {
            analysis: Arc::clone(analysis),
            metrics: None,
            cached: false,
            error: Some(format!("{} q{}: write error: {}", config_key, item.quality, e)),
        };
    }

    let metrics_json = match serde_json::to_string_pretty(&metrics) {
        Ok(j) => j,
        Err(e) => {
            return WorkResult {
                analysis: Arc::clone(analysis),
                metrics: None,
                cached: false,
                error: Some(format!("{} q{}: serialize error: {}", config_key, item.quality, e)),
            };
        }
    };

    if let Err(e) = atomic_write(&item.image_dir.join(&json_name), metrics_json.as_bytes()) {
        return WorkResult {
            analysis: Arc::clone(analysis),
            metrics: None,
            cached: false,
            error: Some(format!("{} q{}: json write error: {}", config_key, item.quality, e)),
        };
    }

    stats.encodings_performed.fetch_add(1, Ordering::Relaxed);

    WorkResult {
        analysis: Arc::clone(analysis),
        metrics: Some(metrics),
        error: None,
        cached: false,
    }
}

/// Measure all metrics for a decoded image (non-cached version).
/// Kept for reference - prefer ImageProcessor::measure() for cached version.
#[allow(dead_code)]
fn measure_metrics(
    original: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
) -> MetricResults {
    let expected_size = width * height * 3;

    if decoded.len() < expected_size {
        return MetricResults {
            butteraugli: f32::MAX,
            ssimulacra2: 0.0,
            dssim: f32::MAX,
            butteraugli_ms: 0,
            ssim2_ms: 0,
            dssim_ms: 0,
        };
    }

    let decoded_slice = &decoded[..expected_size];

    // Calculate all three metrics in parallel using rayon, with timing
    let ((butteraugli, ba_ms), ((ssimulacra2, ssim_ms), (dssim, dssim_ms))) = rayon::join(
        || {
            let start = Instant::now();
            use codec_eval::metrics::butteraugli::calculate_butteraugli;
            let result = calculate_butteraugli(original, decoded_slice, width, height)
                .map(|s| s as f32)
                .unwrap_or(f32::MAX);
            (result, start.elapsed().as_millis() as u64)
        },
        || {
            rayon::join(
                || {
                    let start = Instant::now();
                    // Use fast-ssim2 for faster SIMD-accelerated SSIMULACRA2
                    use fast_ssim2::Ssimulacra2Reference;
                    use imgref::Img;

                    let ref_arr = rgb8_to_array(original);
                    let ref_img = Img::new(ref_arr.as_slice(), width, height);

                    let ssim_ref = match Ssimulacra2Reference::new(ref_img) {
                        Ok(r) => r,
                        Err(_) => return (0.0f32, start.elapsed().as_millis() as u64),
                    };

                    let test_arr = rgb8_to_array(decoded_slice);
                    let test_img = Img::new(test_arr.as_slice(), width, height);

                    let result = ssim_ref.compare(test_img).map(|s| s as f32).unwrap_or(0.0);
                    (result, start.elapsed().as_millis() as u64)
                },
                || {
                    let start = Instant::now();
                    use codec_eval::metrics::dssim::{calculate_dssim, rgb8_to_dssim_image};
                    use codec_eval::viewing::ViewingCondition;
                    let ref_img = rgb8_to_dssim_image(original, width, height);
                    let test_img = rgb8_to_dssim_image(decoded_slice, width, height);
                    let viewing = ViewingCondition::default();
                    let result = calculate_dssim(&ref_img, &test_img, &viewing)
                        .map(|s| s as f32)
                        .unwrap_or(f32::MAX);
                    (result, start.elapsed().as_millis() as u64)
                },
            )
        },
    );

    MetricResults {
        butteraugli,
        ssimulacra2,
        dssim,
        butteraugli_ms: ba_ms,
        ssim2_ms: ssim_ms,
        dssim_ms: dssim_ms,
    }
}

// ============================================================================
// Image Discovery
// ============================================================================

fn find_images(dir: &Path, max: Option<usize>) -> Vec<PathBuf> {
    let mut images = Vec::new();
    let limit = max.unwrap_or(usize::MAX);

    fn scan_dir(dir: &Path, images: &mut Vec<PathBuf>, limit: usize) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if images.len() >= limit {
                    return;
                }
                let path = entry.path();
                if path.is_dir() {
                    scan_dir(&path, images, limit);
                } else if path.extension().map_or(false, |e| e == "png") {
                    images.push(path);
                }
            }
        }
    }

    scan_dir(dir, &mut images, limit);
    images.sort();
    images
}

// ============================================================================
// Aggregated Analysis
// ============================================================================

/// Quality range bucket for grouping results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QualityBucket {
    metric: String,
    range_name: String,
}

/// Analysis result showing which config won for a given bucket
#[derive(Debug, Clone, Serialize)]
struct BucketWinner {
    metric: String,
    range_name: String,
    range_min: f32,
    range_max: f32,
    winner_config: String,
    win_count: usize,
    total_images: usize,
    win_percentage: f32,
    avg_improvement: f32,
}

/// Image characteristics of images where a config won
#[derive(Debug, Clone, Serialize)]
struct WinnerCharacteristics {
    config: String,
    metric: String,
    range_name: String,
    avg_variance: f32,
    avg_edge_density: f32,
    avg_chroma_complexity: f32,
    avg_uniform_blocks: f32,
    image_count: usize,
}

fn run_aggregated_analysis(output_dir: &Path) -> Result<(), String> {
    let csv_path = output_dir.join("results.csv");
    if !csv_path.exists() {
        println!("\nNo results.csv found, skipping analysis.");
        return Ok(());
    }

    println!("\n{:=^70}", " AGGREGATED ANALYSIS ");

    // Load all results
    let file = File::open(&csv_path).map_err(|e| format!("Failed to open CSV: {}", e))?;
    let mut reader = csv::Reader::from_reader(file);

    let all_rows: Vec<CsvRow> = reader.deserialize().filter_map(|r| r.ok()).collect();

    if all_rows.is_empty() {
        println!("No data in results.csv");
        return Ok(());
    }

    // Dedupe rows by (source_hash, config_key, quality), keeping the latest (by timestamp)
    // This handles cases where results.csv has duplicate entries from multiple runs
    let mut deduped: HashMap<(String, String, u8), CsvRow> = HashMap::new();
    for row in all_rows {
        let key = (row.source_hash.clone(), row.config_key.clone(), row.quality);
        deduped
            .entry(key)
            .and_modify(|existing| {
                // Keep the one with newer timestamp (lexicographic comparison works for ISO8601)
                if row.timestamp > existing.timestamp {
                    *existing = row.clone();
                }
            })
            .or_insert(row);
    }
    let rows: Vec<CsvRow> = deduped.into_values().collect();

    println!(
        "Loaded {} unique encoding results (after deduplication)",
        rows.len()
    );

    // Count unique images
    let unique_images: std::collections::HashSet<_> = rows.iter().map(|r| &r.source_hash).collect();
    let unique_configs: std::collections::HashSet<_> = rows.iter().map(|r| &r.config_key).collect();
    println!("Unique images: {}", unique_images.len());
    println!("Unique configs: {:?}", unique_configs);

    // Define quality ranges for each metric
    let bpp_ranges = [
        ("very_low", 0.0, 0.3),
        ("low", 0.3, 0.5),
        ("medium", 0.5, 0.8),
        ("high", 0.8, 1.2),
        ("very_high", 1.2, 3.0),
    ];

    let ssim2_ranges = [
        ("poor", 0.0, 60.0),
        ("acceptable", 60.0, 75.0),
        ("good", 75.0, 85.0),
        ("excellent", 85.0, 95.0),
        ("near_perfect", 95.0, 100.0),
    ];

    let ba_ranges = [
        ("excellent", 0.0, 1.0),
        ("good", 1.0, 2.0),
        ("acceptable", 2.0, 3.0),
        ("noticeable", 3.0, 5.0),
        ("poor", 5.0, 20.0),
    ];

    let dssim_ranges = [
        ("imperceptible", 0.0, 0.0003),
        ("marginal", 0.0003, 0.0007),
        ("subtle", 0.0007, 0.0015),
        ("noticeable", 0.0015, 0.003),
        ("degraded", 0.003, 0.1),
    ];

    println!("\n{:-^70}", " BPP Range Analysis ");
    analyze_by_bpp_range(&rows, &bpp_ranges);

    println!("\n{:-^70}", " SSIMULACRA2 Range Analysis ");
    analyze_metric_winners(&rows, "ssimulacra2", &ssim2_ranges, true);

    println!("\n{:-^70}", " Butteraugli Range Analysis ");
    analyze_metric_winners(&rows, "butteraugli", &ba_ranges, false);

    println!("\n{:-^70}", " DSSIM Range Analysis ");
    analyze_metric_winners(&rows, "dssim", &dssim_ranges, false);

    println!("\n{:-^70}", " Image Characteristic Correlations ");
    analyze_correlations(&rows);

    // Save detailed analysis to text file
    let analysis_path = output_dir.join("analysis_summary.txt");
    save_analysis_txt(
        &rows,
        &analysis_path,
        &bpp_ranges,
        &ssim2_ranges,
        &ba_ranges,
        &dssim_ranges,
    )?;
    println!("\nDetailed analysis saved to {:?}", analysis_path);

    Ok(())
}

fn analyze_by_bpp_range(rows: &[CsvRow], ranges: &[(&str, f32, f32)]) {
    for (name, min, max) in ranges {
        let in_range: Vec<_> = rows
            .iter()
            .filter(|r| r.bpp >= *min && r.bpp < *max)
            .collect();

        if in_range.is_empty() {
            continue;
        }

        // Group by image
        let mut by_image: HashMap<&str, Vec<&CsvRow>> = HashMap::new();
        for row in &in_range {
            by_image.entry(&row.source_hash).or_default().push(row);
        }

        // For each metric, count wins per config
        let mut ssim2_wins: HashMap<&str, usize> = HashMap::new();
        let mut ba_wins: HashMap<&str, usize> = HashMap::new();
        let mut dssim_wins: HashMap<&str, usize> = HashMap::new();

        for (_hash, image_rows) in &by_image {
            // Best SSIM2 (higher is better)
            if let Some(best) = image_rows.iter().max_by(|a, b| {
                a.ssimulacra2
                    .partial_cmp(&b.ssimulacra2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *ssim2_wins.entry(&best.config_key).or_default() += 1;
            }

            // Best Butteraugli (lower is better)
            if let Some(best) = image_rows.iter().min_by(|a, b| {
                a.butteraugli
                    .partial_cmp(&b.butteraugli)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *ba_wins.entry(&best.config_key).or_default() += 1;
            }

            // Best DSSIM (lower is better)
            if let Some(best) = image_rows.iter().min_by(|a, b| {
                a.dssim
                    .partial_cmp(&b.dssim)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *dssim_wins.entry(&best.config_key).or_default() += 1;
            }
        }

        let total = by_image.len();
        println!("\nBPP {}-{} ({} range, {} images):", min, max, name, total);

        print!("  SSIM2 wins:  ");
        for (config, count) in &ssim2_wins {
            print!(
                "{}: {} ({:.0}%)  ",
                config,
                count,
                *count as f32 / total as f32 * 100.0
            );
        }
        println!();

        print!("  BA wins:     ");
        for (config, count) in &ba_wins {
            print!(
                "{}: {} ({:.0}%)  ",
                config,
                count,
                *count as f32 / total as f32 * 100.0
            );
        }
        println!();

        print!("  DSSIM wins:  ");
        for (config, count) in &dssim_wins {
            print!(
                "{}: {} ({:.0}%)  ",
                config,
                count,
                *count as f32 / total as f32 * 100.0
            );
        }
        println!();
    }
}

fn analyze_metric_winners(
    rows: &[CsvRow],
    metric: &str,
    ranges: &[(&str, f32, f32)],
    higher_is_better: bool,
) {
    let get_value = |row: &CsvRow| -> f32 {
        match metric {
            "ssimulacra2" => row.ssimulacra2,
            "butteraugli" => row.butteraugli,
            "dssim" => row.dssim,
            _ => 0.0,
        }
    };

    for (name, min, max) in ranges {
        let in_range: Vec<_> = rows
            .iter()
            .filter(|r| {
                let v = get_value(r);
                v >= *min && v < *max
            })
            .collect();

        if in_range.is_empty() {
            continue;
        }

        // Group by image, then find which config produced best result (smallest file)
        let mut by_image: HashMap<&str, Vec<&CsvRow>> = HashMap::new();
        for row in &in_range {
            by_image.entry(&row.source_hash).or_default().push(row);
        }

        let mut config_wins: HashMap<&str, usize> = HashMap::new();
        let mut config_bpp_sum: HashMap<&str, f32> = HashMap::new();

        for (_hash, image_rows) in &by_image {
            // Best = smallest file (lowest BPP) in this quality range
            if let Some(best) = image_rows.iter().min_by(|a, b| {
                a.bpp
                    .partial_cmp(&b.bpp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *config_wins.entry(&best.config_key).or_default() += 1;
                *config_bpp_sum.entry(&best.config_key).or_default() += best.bpp;
            }
        }

        let total = by_image.len();
        println!(
            "\n{} {} ({} range, {} images at this quality):",
            metric.to_uppercase(),
            name,
            if higher_is_better {
                format!("{:.0}-{:.0}", min, max)
            } else {
                format!("{:.2}-{:.2}", min, max)
            },
            total
        );

        println!("  Best config (smallest file at this quality):");
        let mut sorted: Vec<_> = config_wins.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (config, count) in sorted {
            let avg_bpp = config_bpp_sum.get(config).unwrap_or(&0.0) / *count as f32;
            println!(
                "    {}: {} wins ({:.0}%), avg BPP: {:.3}",
                config,
                count,
                *count as f32 / total as f32 * 100.0,
                avg_bpp
            );
        }
    }
}

fn analyze_correlations(rows: &[CsvRow]) {
    // Group by config and analyze which image characteristics correlate with wins
    let configs: std::collections::HashSet<_> = rows.iter().map(|r| &r.config_key).collect();

    for config in &configs {
        let config_rows: Vec<_> = rows.iter().filter(|r| &r.config_key == *config).collect();

        if config_rows.is_empty() {
            continue;
        }

        // Calculate averages
        let avg_variance =
            config_rows.iter().map(|r| r.variance).sum::<f32>() / config_rows.len() as f32;
        let avg_edge =
            config_rows.iter().map(|r| r.edge_density).sum::<f32>() / config_rows.len() as f32;
        let avg_chroma =
            config_rows.iter().map(|r| r.chroma_complexity).sum::<f32>() / config_rows.len() as f32;
        let avg_uniform = config_rows
            .iter()
            .map(|r| r.uniform_block_fraction)
            .sum::<f32>()
            / config_rows.len() as f32;
        let avg_bpp = config_rows.iter().map(|r| r.bpp).sum::<f32>() / config_rows.len() as f32;
        let avg_ssim2 =
            config_rows.iter().map(|r| r.ssimulacra2).sum::<f32>() / config_rows.len() as f32;
        let avg_ba =
            config_rows.iter().map(|r| r.butteraugli).sum::<f32>() / config_rows.len() as f32;

        println!("\n{}:", config);
        println!("  {} encodings, avg BPP: {:.3}", config_rows.len(), avg_bpp);
        println!("  Avg quality: SSIM2={:.1}, BA={:.2}", avg_ssim2, avg_ba);
        println!(
            "  Image chars: var={:.0}, edge={:.3}, chroma={:.3}, uniform={:.3}",
            avg_variance, avg_edge, avg_chroma, avg_uniform
        );
    }
}

fn save_analysis_txt(
    rows: &[CsvRow],
    path: &Path,
    bpp_ranges: &[(&str, f32, f32)],
    ssim2_ranges: &[(&str, f32, f32)],
    ba_ranges: &[(&str, f32, f32)],
    dssim_ranges: &[(&str, f32, f32)],
) -> Result<(), String> {
    use std::fmt::Write as FmtWrite;

    let mut output = String::new();

    // Header
    writeln!(output, "{:=^80}", " HEURISTIC DISCOVERY ANALYSIS ").unwrap();
    writeln!(
        output,
        "Generated: {}",
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )
    .unwrap();
    writeln!(output).unwrap();

    // Summary stats
    let unique_images: std::collections::HashSet<_> = rows.iter().map(|r| &r.source_hash).collect();
    let mut configs: Vec<_> = rows
        .iter()
        .map(|r| r.config_key.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    configs.sort();

    let bpp_values: Vec<f32> = rows.iter().map(|r| r.bpp).collect();
    let bpp_min = bpp_values.iter().cloned().fold(f32::MAX, f32::min);
    let bpp_max = bpp_values.iter().cloned().fold(f32::MIN, f32::max);
    let bpp_mean = bpp_values.iter().sum::<f32>() / bpp_values.len() as f32;

    writeln!(output, "{:-^80}", " SUMMARY ").unwrap();
    writeln!(output, "Total encodings:  {}", rows.len()).unwrap();
    writeln!(output, "Unique images:    {}", unique_images.len()).unwrap();
    writeln!(output, "Configs tested:   {}", configs.join(", ")).unwrap();
    writeln!(
        output,
        "BPP range:        {:.3} - {:.3} (mean: {:.3})",
        bpp_min, bpp_max, bpp_mean
    )
    .unwrap();
    writeln!(
        output,
        "SSIM2 mean:       {:.2}",
        rows.iter().map(|r| r.ssimulacra2).sum::<f32>() / rows.len() as f32
    )
    .unwrap();
    writeln!(
        output,
        "Butteraugli mean: {:.2}",
        rows.iter().map(|r| r.butteraugli).sum::<f32>() / rows.len() as f32
    )
    .unwrap();
    writeln!(
        output,
        "DSSIM mean:       {:.6}",
        rows.iter().map(|r| r.dssim).sum::<f32>() / rows.len() as f32
    )
    .unwrap();
    writeln!(output).unwrap();

    // BPP Range Analysis
    writeln!(output, "{:-^80}", " BPP RANGE ANALYSIS ").unwrap();
    writeln!(
        output,
        "For each BPP range, shows which config produces best quality metrics."
    )
    .unwrap();
    writeln!(output).unwrap();

    for (name, min, max) in bpp_ranges {
        let in_range: Vec<_> = rows
            .iter()
            .filter(|r| r.bpp >= *min && r.bpp < *max)
            .collect();

        if in_range.is_empty() {
            continue;
        }

        let mut by_image: HashMap<&str, Vec<&CsvRow>> = HashMap::new();
        for row in &in_range {
            by_image.entry(&row.source_hash).or_default().push(row);
        }

        let mut ssim2_wins: HashMap<&str, usize> = HashMap::new();
        let mut ba_wins: HashMap<&str, usize> = HashMap::new();
        let mut dssim_wins: HashMap<&str, usize> = HashMap::new();

        for (_hash, image_rows) in &by_image {
            if let Some(best) = image_rows.iter().max_by(|a, b| {
                a.ssimulacra2
                    .partial_cmp(&b.ssimulacra2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *ssim2_wins.entry(&best.config_key).or_default() += 1;
            }
            if let Some(best) = image_rows.iter().min_by(|a, b| {
                a.butteraugli
                    .partial_cmp(&b.butteraugli)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *ba_wins.entry(&best.config_key).or_default() += 1;
            }
            if let Some(best) = image_rows.iter().min_by(|a, b| {
                a.dssim
                    .partial_cmp(&b.dssim)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *dssim_wins.entry(&best.config_key).or_default() += 1;
            }
        }

        let total = by_image.len();
        writeln!(
            output,
            "BPP {:.1}-{:.1} ({}, {} images):",
            min, max, name, total
        )
        .unwrap();

        let format_wins = |wins: &HashMap<&str, usize>| -> String {
            let mut sorted: Vec<_> = wins.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            sorted
                .iter()
                .map(|(k, v)| format!("{}: {} ({:.0}%)", k, v, **v as f32 / total as f32 * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        };

        writeln!(output, "  SSIM2 best:  {}", format_wins(&ssim2_wins)).unwrap();
        writeln!(output, "  BA best:     {}", format_wins(&ba_wins)).unwrap();
        writeln!(output, "  DSSIM best:  {}", format_wins(&dssim_wins)).unwrap();
        writeln!(output).unwrap();
    }

    // Quality Range Analysis - which config gets smallest file at each quality level
    for (metric_name, ranges, _higher_is_better) in [
        ("SSIMULACRA2", ssim2_ranges, true),
        ("Butteraugli", ba_ranges, false),
        ("DSSIM", dssim_ranges, false),
    ] {
        writeln!(
            output,
            "{:-^80}",
            format!(" {} RANGE ANALYSIS ", metric_name)
        )
        .unwrap();
        writeln!(
            output,
            "For each {} quality level, shows which config produces smallest files.",
            metric_name
        )
        .unwrap();
        writeln!(output).unwrap();

        let get_value = |row: &CsvRow| -> f32 {
            match metric_name {
                "SSIMULACRA2" => row.ssimulacra2,
                "Butteraugli" => row.butteraugli,
                "DSSIM" => row.dssim,
                _ => 0.0,
            }
        };

        for (name, min, max) in ranges.iter() {
            let in_range: Vec<_> = rows
                .iter()
                .filter(|r| {
                    let v = get_value(r);
                    v >= *min && v < *max
                })
                .collect();

            if in_range.is_empty() {
                continue;
            }

            let mut by_image: HashMap<&str, Vec<&CsvRow>> = HashMap::new();
            for row in &in_range {
                by_image.entry(&row.source_hash).or_default().push(row);
            }

            let mut config_wins: HashMap<&str, (usize, f32)> = HashMap::new();
            for (_hash, image_rows) in &by_image {
                if let Some(best) = image_rows.iter().min_by(|a, b| {
                    a.bpp
                        .partial_cmp(&b.bpp)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    let entry = config_wins.entry(&best.config_key).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += best.bpp;
                }
            }

            let total = by_image.len();
            writeln!(
                output,
                "{} {} ({:.4}-{:.4}, {} images):",
                metric_name, name, min, max, total
            )
            .unwrap();

            let mut sorted: Vec<_> = config_wins.iter().collect();
            sorted.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
            for (config, (wins, bpp_sum)) in sorted {
                let avg_bpp = bpp_sum / *wins as f32;
                writeln!(
                    output,
                    "  {}: {} wins ({:.0}%), avg BPP: {:.3}",
                    config,
                    wins,
                    *wins as f32 / total as f32 * 100.0,
                    avg_bpp
                )
                .unwrap();
            }
            writeln!(output).unwrap();
        }
    }

    // Config Characteristics
    writeln!(
        output,
        "{:-^80}",
        " CONFIG PERFORMANCE BY IMAGE CHARACTERISTICS "
    )
    .unwrap();
    writeln!(output).unwrap();

    let mut by_config: HashMap<&str, Vec<&CsvRow>> = HashMap::new();
    for row in rows {
        by_config.entry(&row.config_key).or_default().push(row);
    }

    let mut config_list: Vec<_> = by_config.keys().collect();
    config_list.sort();

    for config in config_list {
        let config_rows = &by_config[config];
        let count = config_rows.len();
        let avg_bpp = config_rows.iter().map(|r| r.bpp).sum::<f32>() / count as f32;
        let avg_ssim2 = config_rows.iter().map(|r| r.ssimulacra2).sum::<f32>() / count as f32;
        let avg_ba = config_rows.iter().map(|r| r.butteraugli).sum::<f32>() / count as f32;
        let avg_dssim = config_rows.iter().map(|r| r.dssim).sum::<f32>() / count as f32;
        let avg_variance = config_rows.iter().map(|r| r.variance).sum::<f32>() / count as f32;
        let avg_edge = config_rows.iter().map(|r| r.edge_density).sum::<f32>() / count as f32;
        let avg_chroma =
            config_rows.iter().map(|r| r.chroma_complexity).sum::<f32>() / count as f32;
        let avg_uniform = config_rows
            .iter()
            .map(|r| r.uniform_block_fraction)
            .sum::<f32>()
            / count as f32;

        writeln!(output, "{}:", config).unwrap();
        writeln!(output, "  Encodings: {}", count).unwrap();
        writeln!(
            output,
            "  Avg BPP: {:.3}, SSIM2: {:.1}, BA: {:.2}, DSSIM: {:.6}",
            avg_bpp, avg_ssim2, avg_ba, avg_dssim
        )
        .unwrap();
        writeln!(
            output,
            "  Image chars: variance={:.0}, edge={:.3}, chroma={:.3}, uniform={:.3}",
            avg_variance, avg_edge, avg_chroma, avg_uniform
        )
        .unwrap();
        writeln!(output).unwrap();
    }

    atomic_write(path, output.as_bytes())
}

fn print_summary(stats: &AtomicRunStats, elapsed: std::time::Duration) {
    let images_processed = stats.images_processed.load(Ordering::Relaxed);
    let images_skipped = stats.images_skipped.load(Ordering::Relaxed);
    let encodings_performed = stats.encodings_performed.load(Ordering::Relaxed);
    let encodings_cached = stats.encodings_cached.load(Ordering::Relaxed);
    let total_encode_time_ms = stats.total_encode_time_ms.load(Ordering::Relaxed);
    let total_metric_time_ms = stats.total_metric_time_ms.load(Ordering::Relaxed);
    let errors = stats.errors.lock().unwrap();

    println!("\n{:=^70}", " RUN SUMMARY ");
    println!("Images processed:    {}", images_processed);
    println!("Images skipped:      {}", images_skipped);
    println!("Encodings performed: {}", encodings_performed);
    println!("Encodings cached:    {}", encodings_cached);
    println!(
        "Total encode time:   {:.1}s",
        total_encode_time_ms as f64 / 1000.0
    );
    println!(
        "Total metric time:   {:.1}s",
        total_metric_time_ms as f64 / 1000.0
    );
    println!("Wall clock time:     {:.1}s", elapsed.as_secs_f64());

    if encodings_performed > 0 {
        let avg_encode = total_encode_time_ms as f64 / encodings_performed as f64;
        let avg_metric = total_metric_time_ms as f64 / encodings_performed as f64;
        println!("Avg encode time:     {:.0}ms", avg_encode);
        println!("Avg metric time:     {:.0}ms", avg_metric);
    }

    if !errors.is_empty() {
        println!("\nErrors ({}):", errors.len());
        for (i, err) in errors.iter().take(10).enumerate() {
            println!("  {}. {}", i + 1, err);
        }
        if errors.len() > 10 {
            println!("  ... and {} more", errors.len() - 10);
        }
    }
}

// ============================================================================
// Work Item Processing (Legacy non-lockstep mode)
// ============================================================================

/// Process a single work item (one encoding at one quality level).
/// This is the legacy non-cached version. Kept for reference.
/// Prefer process_image_lockstep() which uses cached metric references.
#[allow(dead_code)]
fn process_work_item(item: &WorkItem, stats: &AtomicRunStats, args: &Args) -> WorkResult {
    let config_key = item.config.key();

    // Check for existing cache
    let pattern = format!(
        "{}-q{}_v{}.json",
        config_key, item.quality, item.cache_version
    );
    let cached = fs::read_dir(&item.image_dir).ok().and_then(|entries| {
        entries
            .flatten()
            .find(|e| e.file_name().to_string_lossy().ends_with(&pattern))
    });

    if cached.is_some() && !args.force {
        stats.encodings_cached.fetch_add(1, Ordering::Relaxed);
        return WorkResult {
            analysis: item.analysis.clone(),
            metrics: None,
            cached: true,
            error: None,
        };
    }

    // Encode using the Encode trait
    let start = Instant::now();
    let jpeg_data =
        match item
            .config
            .encode(&item.rgb_pixels, item.width, item.height, item.quality)
        {
            Ok(data) => data,
            Err(e) => {
                return WorkResult {
                    analysis: item.analysis.clone(),
                    metrics: None,
                    cached: false,
                    error: Some(format!("{} q{}: {}", config_key, item.quality, e)),
                };
            }
        };
    let encode_time = start.elapsed().as_millis() as u64;
    stats
        .total_encode_time_ms
        .fetch_add(encode_time, Ordering::Relaxed);

    let size_bytes = jpeg_data.len();
    let bpp = (size_bytes * 8) as f32 / (item.width * item.height) as f32;

    // Decode and measure
    let decode_start = Instant::now();
    let decoded = match decode_jpeg(&jpeg_data) {
        Ok(d) => d,
        Err(e) => {
            return WorkResult {
                analysis: item.analysis.clone(),
                metrics: None,
                cached: false,
                error: Some(format!(
                    "{} q{}: decode error: {}",
                    config_key, item.quality, e
                )),
            };
        }
    };
    let decode_ms = decode_start.elapsed().as_millis() as u64;
    stats.total_decode_ms.fetch_add(decode_ms, Ordering::Relaxed);

    let metric_start = Instant::now();
    let metric_results = measure_metrics(&item.rgb_pixels, &decoded, item.width, item.height);
    let butteraugli = metric_results.butteraugli;
    let ssimulacra2 = metric_results.ssimulacra2;
    let dssim = metric_results.dssim;

    // Accumulate per-metric timing
    stats.total_butteraugli_ms.fetch_add(metric_results.butteraugli_ms, Ordering::Relaxed);
    stats.total_ssim2_ms.fetch_add(metric_results.ssim2_ms, Ordering::Relaxed);
    stats.total_dssim_ms.fetch_add(metric_results.dssim_ms, Ordering::Relaxed);
    stats
        .total_metric_time_ms
        .fetch_add(metric_start.elapsed().as_millis() as u64, Ordering::Relaxed);

    // Create metrics
    let metrics = EncodingMetrics {
        source_hash: item.analysis.source_hash.clone(),
        config_key: config_key.clone(),
        quality: item.quality,
        cache_version: item.cache_version,
        size_bytes,
        bpp,
        butteraugli,
        ssimulacra2,
        dssim,
        encode_time_ms: encode_time,
        timestamp: Utc::now(),
    };

    // Write files with metric-based names
    let jpg_name = format_encoding_filename(
        bpp,
        ssimulacra2,
        butteraugli,
        &config_key,
        item.quality,
        item.cache_version,
    );
    let json_name = format_metrics_filename(
        bpp,
        ssimulacra2,
        butteraugli,
        &config_key,
        item.quality,
        item.cache_version,
    );

    if let Err(e) = atomic_write(&item.image_dir.join(&jpg_name), &jpeg_data) {
        return WorkResult {
            analysis: item.analysis.clone(),
            metrics: None,
            cached: false,
            error: Some(format!(
                "{} q{}: write error: {}",
                config_key, item.quality, e
            )),
        };
    }

    let metrics_json = match serde_json::to_string_pretty(&metrics) {
        Ok(j) => j,
        Err(e) => {
            return WorkResult {
                analysis: item.analysis.clone(),
                metrics: None,
                cached: false,
                error: Some(format!(
                    "{} q{}: serialize error: {}",
                    config_key, item.quality, e
                )),
            };
        }
    };

    if let Err(e) = atomic_write(&item.image_dir.join(&json_name), metrics_json.as_bytes()) {
        return WorkResult {
            analysis: item.analysis.clone(),
            metrics: None,
            cached: false,
            error: Some(format!(
                "{} q{}: write metrics error: {}",
                config_key, item.quality, e
            )),
        };
    }

    stats.encodings_performed.fetch_add(1, Ordering::Relaxed);

    WorkResult {
        analysis: item.analysis.clone(),
        metrics: Some(metrics),
        cached: false,
        error: None,
    }
}

/// Prepare an image: load, analyze, create directory
fn prepare_image(
    image_path: &Path,
    output_dir: &Path,
) -> Result<(Arc<Vec<u8>>, usize, usize, Arc<ImageAnalysis>, PathBuf), String> {
    // Load image
    let file = fs::File::open(image_path).map_err(|e| format!("Failed to open image: {}", e))?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("Failed to read PNG info: {}", e))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode PNG: {}", e))?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB
    let rgb_pixels: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf
            .chunks(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => {
            buf.chunks(2).flat_map(|ga| [ga[0], ga[0], ga[0]]).collect()
        }
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type)),
    };

    // Compute source hash
    let source_hash = compute_source_hash(&rgb_pixels);
    let source_name = image_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    // Create image directory
    let image_dir = output_dir.join("images").join(&source_hash);
    fs::create_dir_all(&image_dir).map_err(|e| format!("Failed to create image dir: {}", e))?;

    // Copy original if not present
    let original_path = image_dir.join("original.png");
    if !original_path.exists() {
        fs::copy(image_path, &original_path)
            .map_err(|e| format!("Failed to copy original: {}", e))?;
    }

    // Analyze image
    let analysis = analyze_image(&rgb_pixels, width, height, &source_hash, &source_name);
    let analysis_path = image_dir.join("analysis.json");
    let analysis_json = serde_json::to_string_pretty(&analysis)
        .map_err(|e| format!("Failed to serialize analysis: {}", e))?;
    atomic_write(&analysis_path, analysis_json.as_bytes())?;

    Ok((
        Arc::new(rgb_pixels),
        width,
        height,
        Arc::new(analysis),
        image_dir,
    ))
}

// ============================================================================
// Codec Verification
// ============================================================================

/// Result of verifying a single encoding
#[derive(Debug)]
struct VerifyResult {
    config_key: String,
    quality: u8,
    source_hash: String,
    matched: bool,
    old_size: usize,
    new_size: usize,
    /// Hash of the old JPEG data
    old_hash: String,
    /// Hash of the new JPEG data
    new_hash: String,
}

/// Verify a single encoding by re-encoding and comparing output
fn verify_single_encoding(
    rgb_pixels: &[u8],
    width: usize,
    height: usize,
    config: &Config,
    quality: u8,
    cached_jpeg_path: &Path,
) -> Result<VerifyResult, String> {
    // Read the cached JPEG
    let cached_data = fs::read(cached_jpeg_path)
        .map_err(|e| format!("Failed to read cached JPEG: {}", e))?;

    // Re-encode with current codec
    let new_data = config.encode(rgb_pixels, width, height, quality)?;

    // Compute hashes
    let old_hash = compute_source_hash(&cached_data);
    let new_hash = compute_source_hash(&new_data);

    Ok(VerifyResult {
        config_key: config.key(),
        quality,
        source_hash: String::new(), // filled in by caller
        matched: old_hash == new_hash,
        old_size: cached_data.len(),
        new_size: new_data.len(),
        old_hash,
        new_hash,
    })
}

/// Run quick verification at startup: 3 quality levels per config, 1 image
/// Returns Ok(()) if all pass, Err with details if any fail
fn run_quick_verify(
    output_dir: &Path,
    configs: &[Config],
    manifest: &CacheManifest,
) -> Result<(), Vec<VerifyResult>> {
    println!("\n{:=^70}", " STARTUP VERIFICATION ");

    // Find one cached image directory
    let images_dir = output_dir.join("images");
    if !images_dir.exists() {
        println!("No cached images found, skipping verification.");
        return Ok(());
    }

    let image_dirs: Vec<_> = fs::read_dir(&images_dir)
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .collect()
        })
        .unwrap_or_default();

    if image_dirs.is_empty() {
        println!("No cached images found, skipping verification.");
        return Ok(());
    }

    // Use the first image directory
    let image_dir = image_dirs[0].path();
    let source_hash = image_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    // Load the original image
    let original_path = image_dir.join("original.png");
    if !original_path.exists() {
        println!("No original.png in {:?}, skipping verification.", image_dir);
        return Ok(());
    }

    let (rgb_pixels, width, height) = match load_png(&original_path) {
        Ok(data) => data,
        Err(e) => {
            println!("Failed to load original: {}, skipping verification.", e);
            return Ok(());
        }
    };

    // Quality levels to check
    let quality_levels = [25u8, 50, 75];
    let mut failures = Vec::new();
    let mut checked = 0;

    for config in configs {
        let key = config.key();
        let cache_entry = match manifest.configs.get(&key) {
            Some(e) => e,
            None => continue,
        };

        for &quality in &quality_levels {
            // Find cached JPEG for this config/quality
            let pattern = format!("{}-q{}_v{}.jpg", key, quality, cache_entry.version);
            let cached_jpeg = fs::read_dir(&image_dir)
                .ok()
                .and_then(|entries| {
                    entries
                        .flatten()
                        .find(|e| e.file_name().to_string_lossy().ends_with(&pattern))
                });

            if let Some(entry) = cached_jpeg {
                checked += 1;
                match verify_single_encoding(
                    &rgb_pixels,
                    width,
                    height,
                    config,
                    quality,
                    &entry.path(),
                ) {
                    Ok(mut result) => {
                        result.source_hash = source_hash.clone();
                        if !result.matched {
                            failures.push(result);
                        }
                    }
                    Err(e) => {
                        println!("  {} q{}: verification error: {}", key, quality, e);
                    }
                }
            }
        }
    }

    if failures.is_empty() {
        println!(
            "Quick verification passed: {} encodings checked, all match.",
            checked
        );
        Ok(())
    } else {
        println!(
            "\nWARNING: {} of {} encodings have changed!",
            failures.len(),
            checked
        );
        for f in &failures {
            println!(
                "  {} q{}: size {} -> {} bytes, hash {} -> {}",
                f.config_key, f.quality, f.old_size, f.new_size, f.old_hash, f.new_hash
            );
        }
        println!("\nCodec output has changed. You need to increment the version number");
        println!("in Config::version_info() for the affected configs.");
        Err(failures)
    }
}

/// Run full verification: check all cached encodings
fn run_full_verify(output_dir: &Path, configs: &[Config], manifest: &CacheManifest) {
    println!("\n{:=^70}", " FULL VERIFICATION ");

    let images_dir = output_dir.join("images");
    if !images_dir.exists() {
        println!("No cached images found.");
        return;
    }

    let mut total_checked = 0;
    let mut total_failures = 0;
    let mut failures: Vec<VerifyResult> = Vec::new();

    let image_dirs: Vec<_> = fs::read_dir(&images_dir)
        .ok()
        .map(|entries| entries.flatten().filter(|e| e.path().is_dir()).collect())
        .unwrap_or_default();

    println!("Verifying {} cached images...", image_dirs.len());

    for (idx, image_entry) in image_dirs.iter().enumerate() {
        let image_dir = image_entry.path();
        let source_hash = image_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        // Load the original image
        let original_path = image_dir.join("original.png");
        if !original_path.exists() {
            continue;
        }

        let (rgb_pixels, width, height) = match load_png(&original_path) {
            Ok(data) => data,
            Err(_) => continue,
        };

        let mut image_failures = 0;

        for config in configs {
            let key = config.key();
            let cache_entry = match manifest.configs.get(&key) {
                Some(e) => e,
                None => continue,
            };

            // Find all cached JPEGs for this config at current version
            // Filename format: {bpp}bpp_{ssim2}ss_{ba}ba_{config}-q{quality}_v{version}.jpg
            let config_pattern = format!("_{}-q", key);
            let version_pattern = format!("_v{}.jpg", cache_entry.version);
            let cached_jpegs: Vec<_> = fs::read_dir(&image_dir)
                .ok()
                .map(|entries| {
                    entries
                        .flatten()
                        .filter(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            name.contains(&config_pattern) && name.ends_with(&version_pattern)
                        })
                        .collect()
                })
                .unwrap_or_default();

            for entry in cached_jpegs {
                // Extract quality from filename (format: {bpp}bpp_{ssim2}ss_{ba}ba_{config}-q{quality}_v{version}.jpg)
                let name = entry.file_name().to_string_lossy().to_string();
                let quality = extract_quality_from_filename(&name);
                if quality == 0 {
                    continue;
                }

                total_checked += 1;

                match verify_single_encoding(
                    &rgb_pixels,
                    width,
                    height,
                    config,
                    quality,
                    &entry.path(),
                ) {
                    Ok(mut result) => {
                        result.source_hash = source_hash.clone();
                        if !result.matched {
                            image_failures += 1;
                            total_failures += 1;
                            failures.push(result);
                        }
                    }
                    Err(_) => {}
                }
            }
        }

        if (idx + 1) % 10 == 0 || idx == image_dirs.len() - 1 {
            print!(
                "\r[{}/{}] {} failures so far...    ",
                idx + 1,
                image_dirs.len(),
                total_failures
            );
            io::stdout().flush().ok();
        }
    }

    println!();
    println!("\n{:-^70}", " VERIFICATION RESULTS ");
    println!("Total encodings checked: {}", total_checked);
    println!("Failures (output changed): {}", total_failures);

    if !failures.is_empty() {
        // Group by config
        let mut by_config: HashMap<String, Vec<&VerifyResult>> = HashMap::new();
        for f in &failures {
            by_config.entry(f.config_key.clone()).or_default().push(f);
        }

        println!("\nFailures by config:");
        for (config, config_failures) in &by_config {
            println!("  {}: {} failures", config, config_failures.len());
            // Show first few examples
            for f in config_failures.iter().take(3) {
                println!(
                    "    - {} q{}: {} -> {} bytes",
                    f.source_hash, f.quality, f.old_size, f.new_size
                );
            }
            if config_failures.len() > 3 {
                println!("    ... and {} more", config_failures.len() - 3);
            }
        }

        println!("\nCodec output has changed. Update Config::version_info() to increment");
        println!("the version for affected configs, then re-run the benchmark.");
    } else {
        println!("\nAll encodings match! Codec outputs are consistent.");
    }
}

/// Extract quality level from encoding filename
fn extract_quality_from_filename(filename: &str) -> u8 {
    // Format: {bpp}bpp_{ssim2}ss_{ba}ba_{config}-q{quality}_v{version}.jpg
    if let Some(q_pos) = filename.rfind("-q") {
        let after_q = &filename[q_pos + 2..];
        if let Some(underscore_pos) = after_q.find('_') {
            if let Ok(q) = after_q[..underscore_pos].parse::<u8>() {
                return q;
            }
        }
    }
    0
}

/// Load PNG and return (RGB pixels, width, height)
fn load_png(path: &Path) -> Result<(Vec<u8>, usize, usize), String> {
    let file = fs::File::open(path).map_err(|e| format!("Failed to open: {}", e))?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("Failed to read PNG info: {}", e))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode PNG: {}", e))?;

    let width = info.width as usize;
    let height = info.height as usize;

    let rgb_pixels: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf
            .chunks(4)
            .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => {
            buf.chunks(2).flat_map(|ga| [ga[0], ga[0], ga[0]]).collect()
        }
        _ => return Err(format!("Unsupported color type: {:?}", info.color_type)),
    };

    Ok((rgb_pixels, width, height))
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args = Args::parse();

    // Initialize CUDA if --gpu flag is set
    #[cfg(feature = "gpu")]
    if args.gpu {
        if init_cuda_once() {
            println!("GPU acceleration enabled (CUDA initialized)");
        } else {
            eprintln!("Error: --gpu requested but CUDA initialization failed");
            eprintln!("Make sure CUDA_PATH is set and CUDA is properly installed");
            std::process::exit(1);
        }
    }

    #[cfg(not(feature = "gpu"))]
    if args.gpu {
        eprintln!("Error: --gpu requested but GPU feature is not enabled");
        eprintln!("Rebuild with: cargo run --release --features gpu --example discover_heuristics");
        std::process::exit(1);
    }

    // Validate args
    if !args.corpus.exists() {
        eprintln!("Corpus directory does not exist: {:?}", args.corpus);
        std::process::exit(1);
    }

    // Create output directory
    fs::create_dir_all(&args.output).expect("Failed to create output directory");
    fs::create_dir_all(args.output.join("images")).expect("Failed to create images directory");

    // Load/create manifest
    let mut manifest = load_or_create_manifest(&args.output).expect("Failed to load manifest");

    // Get configs to test
    let configs = Config::test_subset();

    // Validate manifest against current code
    if let Err(e) = validate_or_update_manifest(&mut manifest, &configs, &args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    // Save updated manifest
    save_manifest(&manifest, &args.output).expect("Failed to save manifest");

    // Handle verification modes
    if args.verify {
        // Full verification mode - check all cached encodings and exit
        run_full_verify(&args.output, &configs, &manifest);
        return;
    }

    // Startup quick-check (unless --skip-verify)
    if !args.skip_verify {
        if let Err(failures) = run_quick_verify(&args.output, &configs, &manifest) {
            eprintln!(
                "\nStartup verification failed: {} encodings have changed.",
                failures.len()
            );
            eprintln!("Use --skip-verify to bypass this check, or update version numbers.");
            std::process::exit(1);
        }
    }

    // Find images
    let images = find_images(&args.corpus, args.max_images);
    println!("Found {} images in {:?}", images.len(), args.corpus);

    if images.is_empty() {
        eprintln!("No PNG images found!");
        std::process::exit(1);
    }

    // Open CSV (append mode) with mutex for thread-safe writes
    let csv_path = args.output.join("results.csv");
    let csv_exists = csv_path.exists();
    let csv_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)
        .expect("Failed to open CSV");
    let csv_writer = Arc::new(Mutex::new(
        csv::WriterBuilder::new()
            .has_headers(!csv_exists)
            .from_writer(BufWriter::new(csv_file)),
    ));

    let stats = Arc::new(AtomicRunStats::new());
    let start = Instant::now();

    // Phase 1: Prepare all images (load, analyze, create dirs)
    println!("Phase 1: Loading and analyzing {} images...", images.len());
    let prepared: Vec<_> = images
        .iter()
        .filter_map(|image_path| match prepare_image(image_path, &args.output) {
            Ok((rgb_pixels, width, height, analysis, image_dir)) => Some((
                image_path.clone(),
                rgb_pixels,
                width,
                height,
                analysis,
                image_dir,
            )),
            Err(e) => {
                eprintln!(
                    "Error preparing {:?}: {}",
                    image_path.file_name().unwrap_or_default(),
                    e
                );
                stats.add_error(e);
                None
            }
        })
        .collect();

    stats
        .images_processed
        .store(prepared.len(), Ordering::Relaxed);
    println!("Prepared {} images successfully.", prepared.len());

    // Phase 2: Generate all work items
    println!("Phase 2: Generating work items...");
    let mut work_items = Vec::new();

    for (image_path, rgb_pixels, width, height, analysis, image_dir) in &prepared {
        for config in &configs {
            let key = config.key();
            let cache_entry = match manifest.configs.get(&key) {
                Some(e) => e,
                None => {
                    stats.add_error(format!("Config {} not in manifest", key));
                    continue;
                }
            };

            // Generate quality levels (1 to 100 with step)
            let mut q = 1u8;
            while q <= 100 {
                work_items.push(WorkItem {
                    image_path: image_path.clone(),
                    rgb_pixels: rgb_pixels.clone(),
                    width: *width,
                    height: *height,
                    config: *config,
                    quality: q,
                    analysis: analysis.clone(),
                    image_dir: image_dir.clone(),
                    cache_version: cache_entry.version,
                });
                q = q.saturating_add(args.step);
            }
        }
    }

    println!(
        "Generated {} work items across {} images x {} configs.",
        work_items.len(),
        prepared.len(),
        configs.len()
    );

    // Phase 3: Process images in lockstep mode
    //
    // Lockstep processing: For each source image, we create an ImageProcessor that
    // caches the reference image data for butteraugli and SSIM2. This provides
    // ~40-50% speedup for metric calculation since the expensive reference preprocessing
    // only happens once per source image instead of once per encoding.
    //
    // We parallelize across source images (outer parallelism), while processing
    // all configs/qualities for each image sequentially (inner loop) to benefit
    // from the cached references.
    if args.gpu {
        println!("Phase 3: Processing encodings in lockstep mode (GPU SSIM2, sequential)...");
    } else {
        println!("Phase 3: Processing encodings in lockstep mode (cached reference metrics)...");
    }

    // Group work items by source image path
    let mut work_by_image: HashMap<PathBuf, Vec<WorkItem>> = HashMap::new();
    for item in work_items {
        work_by_image
            .entry(item.image_path.clone())
            .or_default()
            .push(item);
    }

    println!(
        "Processing {} images with {} work items each (avg){}",
        work_by_image.len(),
        work_by_image.values().map(|v| v.len()).sum::<usize>() / work_by_image.len().max(1),
        if args.gpu { " [GPU mode - sequential to maintain CUDA context]" } else { "" }
    );

    let args_ref = &args;
    let stats_ref = &stats;
    let csv_writer_ref = &csv_writer;

    // Helper to process one image and write results to CSV
    let process_one_image = |image_path: &PathBuf, items: &[WorkItem]| -> Vec<WorkResult> {
        if args_ref.verbose {
            eprintln!("Processing {:?} ({} encodings)", image_path.file_name().unwrap_or_default(), items.len());
        }

        let image_results = process_image_lockstep(items, stats_ref, args_ref);

        // Write results to CSV
        for result in &image_results {
            if let Some(ref metrics) = result.metrics {
                if metrics.bpp >= args_ref.min_bpp && metrics.bpp <= args_ref.max_bpp {
                    let row = CsvRow {
                        source_hash: result.analysis.source_hash.clone(),
                        source_name: result.analysis.source_name.clone(),
                        width: result.analysis.width,
                        height: result.analysis.height,
                        variance: result.analysis.variance,
                        edge_density: result.analysis.edge_density,
                        chroma_complexity: result.analysis.chroma_complexity,
                        uniform_block_fraction: result.analysis.uniform_block_fraction,
                        config_key: metrics.config_key.clone(),
                        quality: metrics.quality,
                        cache_version: metrics.cache_version,
                        size_bytes: metrics.size_bytes,
                        bpp: metrics.bpp,
                        butteraugli: metrics.butteraugli,
                        ssimulacra2: metrics.ssimulacra2,
                        dssim: metrics.dssim,
                        encode_time_ms: metrics.encode_time_ms,
                        timestamp: metrics.timestamp.to_rfc3339(),
                    };

                    let mut writer = csv_writer_ref.lock().unwrap();
                    if let Err(e) = writer.serialize(&row) {
                        stats_ref.add_error(format!("CSV write error: {}", e));
                    }
                }
            }

            if let Some(ref error) = result.error {
                stats_ref.add_error(error.clone());
            }
        }

        image_results
    };

    // Process images - sequentially for GPU mode (CUDA context is thread-local),
    // parallel for CPU mode
    let results: Vec<WorkResult> = if args.gpu {
        // GPU mode: sequential processing to maintain CUDA context
        work_by_image
            .iter()
            .flat_map(|(image_path, items)| process_one_image(image_path, items))
            .collect()
    } else {
        // CPU mode: parallel processing across images
        work_by_image
            .into_par_iter()
            .flat_map(|(image_path, items)| process_one_image(&image_path, &items))
            .collect()
    };

    // Flush CSV
    {
        let mut writer = csv_writer.lock().unwrap();
        writer.flush().expect("Failed to flush CSV");
    }

    // Print summary
    print_summary(&stats, start.elapsed());

    // Print timing breakdown to identify bottlenecks
    stats.print_timing_breakdown();

    // Count results
    let new_encodings = results.iter().filter(|r| r.metrics.is_some()).count();
    let cached = results.iter().filter(|r| r.cached).count();
    let errors = results.iter().filter(|r| r.error.is_some()).count();
    println!(
        "\nResults: {} new, {} cached, {} errors",
        new_encodings, cached, errors
    );

    // Run aggregated analysis on all accumulated data
    if let Err(e) = run_aggregated_analysis(&args.output) {
        eprintln!("Analysis error: {}", e);
    }

    println!("\nResults written to {:?}", args.output);

    // GPU mode: exit immediately to avoid CUDA cleanup crash
    // The CUDA driver cleanup can cause segfaults on some systems
    #[cfg(feature = "gpu")]
    if args.gpu {
        std::process::exit(0);
    }
}
