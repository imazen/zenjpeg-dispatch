// SPDX-License-Identifier: MIT OR Apache-2.0
//! Runtime-configurable adaptive encoding features.
//!
//! This module provides a unified configuration system for testing and tuning
//! different encoding strategies. The goal is to find optimal heuristics for
//! the Pareto curve regardless of image characteristics.
//!
//! # Feature Categories
//!
//! ## Subsampling Features
//! - `evalchroma`: Content-aware subsampling (upgrades to 4:4:4 for chroma-heavy images)
//! - `fixed_subsampling`: Force specific subsampling mode
//!
//! ## Quantization Features
//! - `trellis`: Rate-distortion optimized coefficient selection (mozjpeg)
//! - `adaptive_quant`: Per-block quantization strength (jpegli)
//! - `hybrid_trellis`: Combine trellis with AQ (jpegli experimental)
//! - `zero_bias`: Perceptual zero-biasing thresholds (jpegli)
//!
//! ## Color Space Features
//! - `ycbcr`: Standard JPEG YCbCr
//! - `xyb`: Perceptual XYB color space (jpegli)
//!
//! ## Entropy Coding Features
//! - `optimize_huffman`: 2-pass Huffman table optimization
//! - `progressive`: Multi-scan progressive encoding
//!
//! ## Post-Processing Features
//! - `deringing`: Reduce ringing artifacts at edges (mozjpeg)
//! - `uniform_block_detection`: Fast-path for solid color blocks

use crate::types::Subsampling;

/// Complete configuration for adaptive encoding.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Core encoding parameters
    pub quality: u8,

    /// Subsampling strategy
    pub subsampling: SubsamplingConfig,

    /// Quantization features
    pub quantization: QuantizationConfig,

    /// Color space selection
    pub color_space: ColorSpaceConfig,

    /// Entropy coding options
    pub entropy: EntropyConfig,

    /// Post-processing options
    pub post_processing: PostProcessingConfig,

    /// Encoder backend selection
    pub backend: EncoderBackend,
}

/// Subsampling configuration
#[derive(Debug, Clone)]
pub enum SubsamplingConfig {
    /// Analyze image with evalchroma and pick optimal subsampling
    Adaptive {
        /// Maximum allowed subsampling (floor, won't go lower than this quality)
        max_subsampling: Subsampling,
    },
    /// Force specific subsampling mode
    Fixed(Subsampling),
}

/// Quantization feature configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Enable trellis quantization (mozjpeg-style RD optimization)
    pub trellis: TrellisMode,

    /// Enable adaptive quantization (jpegli-style per-block adjustment)
    pub adaptive_quant: AdaptiveQuantMode,

    /// Enable perceptual zero-biasing (jpegli)
    pub zero_bias: bool,

    /// Uniform block detection for fast-path encoding
    pub uniform_block_detection: bool,
}

/// Trellis quantization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrellisMode {
    /// No trellis (fastest)
    Disabled,
    /// AC coefficients only (default mozjpeg)
    AcOnly,
    /// AC + DC coefficients (slower, slightly better)
    AcAndDc,
}

/// Adaptive quantization modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveQuantMode {
    /// No adaptive quantization
    Disabled,
    /// Global AQ strength
    Global { strength: u8 },
    /// Per-block AQ based on local variance (jpegli default)
    PerBlock,
    /// Hybrid: AQ + trellis (experimental)
    HybridTrellis,
}

/// Color space configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpaceConfig {
    /// Standard JPEG YCbCr (BT.601)
    YCbCr,
    /// Perceptual XYB color space (jpegli)
    XYB,
}

/// Entropy coding configuration
#[derive(Debug, Clone)]
pub struct EntropyConfig {
    /// Enable 2-pass Huffman table optimization
    pub optimize_huffman: bool,

    /// Progressive encoding level
    pub progressive: ProgressiveMode,
}

/// Progressive encoding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressiveMode {
    /// Sequential (baseline) JPEG
    Sequential,
    /// Standard progressive (10 scans)
    Standard,
    /// Aggressive progressive (more scans, smaller files, slower)
    Aggressive,
}

/// Post-processing configuration
#[derive(Debug, Clone)]
pub struct PostProcessingConfig {
    /// Enable overshoot deringing (mozjpeg)
    pub deringing: bool,

    /// Smoothing filter strength (0 = disabled, 100 = max)
    pub smoothing: u8,
}

/// Encoder backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderBackend {
    /// Auto-select based on quality and features
    Auto,
    /// Force mozjpeg-oxide backend
    Mozjpeg,
    /// Force jpegli backend
    Jpegli,
    /// Try both and pick smaller (slow but optimal)
    BestOf,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            quality: 85,
            subsampling: SubsamplingConfig::Adaptive {
                max_subsampling: Subsampling::S420,
            },
            quantization: QuantizationConfig {
                trellis: TrellisMode::AcOnly,
                adaptive_quant: AdaptiveQuantMode::PerBlock,
                zero_bias: true,
                uniform_block_detection: false,
            },
            color_space: ColorSpaceConfig::YCbCr,
            entropy: EntropyConfig {
                optimize_huffman: true,
                progressive: ProgressiveMode::Sequential,
            },
            post_processing: PostProcessingConfig {
                deringing: true,
                smoothing: 0,
            },
            backend: EncoderBackend::Auto,
        }
    }
}

impl AdaptiveConfig {
    /// Create config optimized for maximum compression
    pub fn max_compression() -> Self {
        Self {
            quality: 75,
            subsampling: SubsamplingConfig::Fixed(Subsampling::S420),
            quantization: QuantizationConfig {
                trellis: TrellisMode::AcAndDc,
                adaptive_quant: AdaptiveQuantMode::PerBlock,
                zero_bias: true,
                uniform_block_detection: true,
            },
            color_space: ColorSpaceConfig::YCbCr,
            entropy: EntropyConfig {
                optimize_huffman: true,
                progressive: ProgressiveMode::Aggressive,
            },
            post_processing: PostProcessingConfig {
                deringing: true,
                smoothing: 0,
            },
            backend: EncoderBackend::BestOf,
        }
    }

    /// Create config optimized for maximum quality
    pub fn max_quality() -> Self {
        Self {
            quality: 95,
            subsampling: SubsamplingConfig::Adaptive {
                max_subsampling: Subsampling::S444,
            },
            quantization: QuantizationConfig {
                trellis: TrellisMode::Disabled,
                adaptive_quant: AdaptiveQuantMode::PerBlock,
                zero_bias: true,
                uniform_block_detection: false,
            },
            color_space: ColorSpaceConfig::YCbCr,
            entropy: EntropyConfig {
                optimize_huffman: true,
                progressive: ProgressiveMode::Sequential,
            },
            post_processing: PostProcessingConfig {
                deringing: false,
                smoothing: 0,
            },
            backend: EncoderBackend::Jpegli,
        }
    }

    /// Create config for fastest encoding
    pub fn fastest() -> Self {
        Self {
            quality: 85,
            subsampling: SubsamplingConfig::Fixed(Subsampling::S420),
            quantization: QuantizationConfig {
                trellis: TrellisMode::Disabled,
                adaptive_quant: AdaptiveQuantMode::Disabled,
                zero_bias: false,
                uniform_block_detection: false,
            },
            color_space: ColorSpaceConfig::YCbCr,
            entropy: EntropyConfig {
                optimize_huffman: false,
                progressive: ProgressiveMode::Sequential,
            },
            post_processing: PostProcessingConfig {
                deringing: false,
                smoothing: 0,
            },
            backend: EncoderBackend::Mozjpeg,
        }
    }

    /// Create config that matches jpegli defaults
    pub fn jpegli_default() -> Self {
        Self {
            quality: 90,
            subsampling: SubsamplingConfig::Fixed(Subsampling::S444),
            quantization: QuantizationConfig {
                trellis: TrellisMode::Disabled,
                adaptive_quant: AdaptiveQuantMode::PerBlock,
                zero_bias: true,
                uniform_block_detection: false,
            },
            color_space: ColorSpaceConfig::YCbCr,
            entropy: EntropyConfig {
                optimize_huffman: true,
                progressive: ProgressiveMode::Sequential,
            },
            post_processing: PostProcessingConfig {
                deringing: false,
                smoothing: 0,
            },
            backend: EncoderBackend::Jpegli,
        }
    }

    /// Create config that matches mozjpeg defaults
    pub fn mozjpeg_default() -> Self {
        Self {
            quality: 85,
            subsampling: SubsamplingConfig::Fixed(Subsampling::S420),
            quantization: QuantizationConfig {
                trellis: TrellisMode::AcOnly,
                adaptive_quant: AdaptiveQuantMode::Disabled,
                zero_bias: false,
                uniform_block_detection: false,
            },
            color_space: ColorSpaceConfig::YCbCr,
            entropy: EntropyConfig {
                optimize_huffman: true,
                progressive: ProgressiveMode::Standard,
            },
            post_processing: PostProcessingConfig {
                deringing: true,
                smoothing: 0,
            },
            backend: EncoderBackend::Mozjpeg,
        }
    }

    // Builder methods

    pub fn quality(mut self, q: u8) -> Self {
        self.quality = q;
        self
    }

    pub fn subsampling(mut self, config: SubsamplingConfig) -> Self {
        self.subsampling = config;
        self
    }

    pub fn trellis(mut self, mode: TrellisMode) -> Self {
        self.quantization.trellis = mode;
        self
    }

    pub fn adaptive_quant(mut self, mode: AdaptiveQuantMode) -> Self {
        self.quantization.adaptive_quant = mode;
        self
    }

    pub fn evalchroma(mut self, enable: bool) -> Self {
        if enable {
            self.subsampling = SubsamplingConfig::Adaptive {
                max_subsampling: Subsampling::S420,
            };
        } else {
            self.subsampling = SubsamplingConfig::Fixed(Subsampling::S420);
        }
        self
    }

    pub fn progressive(mut self, mode: ProgressiveMode) -> Self {
        self.entropy.progressive = mode;
        self
    }

    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.entropy.optimize_huffman = enable;
        self
    }

    pub fn backend(mut self, backend: EncoderBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn deringing(mut self, enable: bool) -> Self {
        self.post_processing.deringing = enable;
        self
    }

    /// Check if this config requires jpegli backend
    pub fn requires_jpegli(&self) -> bool {
        matches!(self.color_space, ColorSpaceConfig::XYB)
            || matches!(self.quantization.adaptive_quant, AdaptiveQuantMode::PerBlock | AdaptiveQuantMode::HybridTrellis)
    }

    /// Check if this config requires mozjpeg backend
    pub fn requires_mozjpeg(&self) -> bool {
        !matches!(self.quantization.trellis, TrellisMode::Disabled)
            || self.post_processing.deringing
    }

    /// Get the effective backend based on config requirements
    pub fn effective_backend(&self) -> EncoderBackend {
        match self.backend {
            EncoderBackend::Auto => {
                if self.requires_jpegli() && !self.requires_mozjpeg() {
                    EncoderBackend::Jpegli
                } else if self.requires_mozjpeg() && !self.requires_jpegli() {
                    EncoderBackend::Mozjpeg
                } else {
                    // Both have required features, or neither does
                    // Default to jpegli at high quality, mozjpeg at low quality
                    if self.quality >= 70 {
                        EncoderBackend::Jpegli
                    } else {
                        EncoderBackend::Mozjpeg
                    }
                }
            }
            other => other,
        }
    }
}

/// Result of encoding with a specific configuration
#[derive(Debug, Clone)]
pub struct EncodingResult {
    /// Encoded JPEG data
    pub data: Vec<u8>,
    /// File size in bytes
    pub size: usize,
    /// Bits per pixel
    pub bpp: f32,
    /// Which backend was used
    pub backend_used: EncoderBackend,
    /// Configuration that was used
    pub config: AdaptiveConfig,
}

/// Benchmark result comparing multiple configurations
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Results for each tested configuration
    pub results: Vec<EncodingResult>,
    /// Index of the Pareto-optimal result (smallest size at best quality)
    pub pareto_best: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AdaptiveConfig::default();
        assert_eq!(config.quality, 85);
        assert!(config.entropy.optimize_huffman);
    }

    #[test]
    fn test_presets() {
        let max_comp = AdaptiveConfig::max_compression();
        assert!(matches!(max_comp.quantization.trellis, TrellisMode::AcAndDc));

        let max_qual = AdaptiveConfig::max_quality();
        assert!(matches!(max_qual.quantization.trellis, TrellisMode::Disabled));

        let fastest = AdaptiveConfig::fastest();
        assert!(!fastest.entropy.optimize_huffman);
    }

    #[test]
    fn test_effective_backend() {
        // High quality prefers jpegli
        let high_q = AdaptiveConfig::default().quality(90);
        assert!(matches!(high_q.effective_backend(), EncoderBackend::Jpegli));

        // Trellis requires mozjpeg
        let trellis = AdaptiveConfig::default()
            .trellis(TrellisMode::AcAndDc)
            .adaptive_quant(AdaptiveQuantMode::Disabled);
        assert!(matches!(trellis.effective_backend(), EncoderBackend::Mozjpeg));

        // Explicit backend overrides auto
        let explicit = AdaptiveConfig::default().backend(EncoderBackend::Mozjpeg);
        assert!(matches!(explicit.effective_backend(), EncoderBackend::Mozjpeg));
    }

    #[test]
    fn test_builder_chain() {
        let config = AdaptiveConfig::default()
            .quality(75)
            .evalchroma(true)
            .trellis(TrellisMode::AcOnly)
            .progressive(ProgressiveMode::Standard)
            .backend(EncoderBackend::BestOf);

        assert_eq!(config.quality, 75);
        assert!(matches!(config.subsampling, SubsamplingConfig::Adaptive { .. }));
        assert!(matches!(config.entropy.progressive, ProgressiveMode::Standard));
    }
}
