# zenjpeg Development Guide

## Project Overview

zenjpeg is a high-quality JPEG encoder that combines the best techniques from mozjpeg and jpegli to achieve Pareto-optimal compression at both low and high quality settings.

**Key insight**: mozjpeg's trellis quantization excels at low quality (Q < 70), while jpegli's adaptive quantization excels at high quality (Q >= 70). zenjpeg automatically selects the best strategy.

## Quick Start

```bash
cd /home/lilith/work/zenjpeg
cargo test          # Run tests
cargo build         # Build library

# Run heuristic discovery benchmark (CPU)
cargo run --release --example discover_heuristics -- \
  --corpus ~/work/codec-corpus/kodak --output /tmp/results

# Run with GPU-accelerated SSIM2 (requires CUDA)
CUDA_PATH=/usr/local/cuda-12.6 cargo run --release --features gpu \
  --example discover_heuristics -- \
  --corpus ~/work/codec-corpus/kodak --output /tmp/results --gpu
```

## Architecture

```
zenjpeg/
├── src/
│   ├── lib.rs              # Public API re-exports
│   ├── error.rs            # Error types (#[non_exhaustive])
│   ├── types.rs            # Core types (Quality, Subsampling, ColorSpace)
│   ├── consts.rs           # Constants, tables, JPEG markers
│   ├── color.rs            # RGB→YCbCr conversion
│   ├── dct.rs              # Forward DCT
│   ├── quant.rs            # Quantization tables
│   ├── huffman.rs          # Huffman table handling
│   ├── entropy.rs          # Entropy encoding (bitstream)
│   ├── trellis.rs          # Trellis quantization (from mozjpeg)
│   ├── adaptive_quant.rs   # Adaptive quantization (from jpegli)
│   ├── strategy.rs         # Encoding strategy selection
│   └── encode.rs           # Main Encoder API
├── examples/
│   ├── discover_heuristics.rs  # Main benchmarking tool (GPU support)
│   ├── pareto_benchmark.rs     # Pareto comparison
│   └── ...
├── tests/                  # Integration tests
├── ARCHITECTURE.md         # Detailed architecture docs
├── RESEARCH.md             # Research findings and decisions
└── CLAUDE.md               # This file
```

## Current Status (Jan 2026)

### Recently Completed
- [x] **GPU-accelerated DSSIM** in discover_heuristics (~16% faster)
- [x] **GPU-accelerated SSIMULACRA2** in discover_heuristics (~14x faster)
- [x] **Lockstep processing mode** with cached metric references
- [x] **ButteraugliReference caching** for ~45% speedup
- [x] **Ssimulacra2Reference caching** for CPU mode
- [x] Sequential processing for GPU mode (CUDA context thread-locality)
- [x] Proper CUDA cleanup with Drop impl and exit(0) workaround

### Benchmarking Infrastructure

**discover_heuristics.rs** - Main benchmarking tool:
- 8 codec configurations: mozjpeg-420/444, mozjpeg-max-420/444, jpegli-420/444, cmozjpeg-420, cmozjpeg-max-420
- 100 quality levels per config
- 3 quality metrics: SSIMULACRA2, Butteraugli, DSSIM
- GPU mode: `--gpu` flag accelerates both SSIM2 and DSSIM
- Lockstep mode: processes one image through all configs before moving to next

**GPU Performance (with --gpu flag):**
```
SSIMULACRA2:    1.8s  (  1.7%)   # GPU-accelerated (~14x faster)
DSSIM:         19.5s  ( 18.5%)   # GPU-accelerated (~16% faster)
Butteraugli:   62.9s  ( 59.5%)   # CPU with cached reference
```

### Codec Selection by Metric
Based on benchmarks (336 images × 6 configs × 100 quality levels):

| Metric | Best Codec | Mean Regret | Notes |
|--------|------------|-------------|-------|
| **SSIMULACRA2** | jpegli-420 | 3.87% | Dominates at all quality levels |
| **Butteraugli** | jpegli-420 | 4.93% | Strongly dominates (50-60% wins) |
| **DSSIM** | mozjpeg-max-420 | **1.01%** | Progressive encoding helps! |

### zenjpeg Encoder Performance
At SSIM2 >= 80 quality target:
- jpegli: **1.310 bpp** (best efficiency)
- mozjpeg-oxide: 1.437 bpp
- **zenjpeg: 1.458 bpp** (only 1.5% larger than mozjpeg-oxide!)

## Pending Work

### High Priority
- [ ] Port jpegli's perceptual adaptive quantization (complex algorithm)
- [ ] Investigate jpegli vs mozjpeg quality gap at high quality
- [ ] Add more image corpora to benchmarks (CID22, etc.)

### Medium Priority
- [ ] Port SIMD DCT from mozjpeg-rs
- [ ] GPU-accelerate Butteraugli (biggest remaining bottleneck at 58%)
- [ ] Add parallel image processing for CPU mode

### Low Priority
- [ ] Design custom SA tables optimized for SSIMULACRA2/Butteraugli
- [ ] Add validation mode to verify cached encodings

## Key Design Decisions

### 1. Strategy Selection
- Q < 50: Use mozjpeg strategy (trellis + progressive)
- Q 50-70: Use hybrid strategy (both trellis + AQ with reduced strength)
- Q >= 70: Use jpegli strategy (adaptive quantization)

### 2. Quality Modes
```rust
pub enum Quality {
    Standard(u8),      // 1-100, auto-selects strategy
    Low(u8),           // Forces mozjpeg strategy
    High(u8),          // Forces jpegli strategy
    Perceptual(f32),   // Target SSIMULACRA2 score
    TargetSize(usize), // Binary search for quality
}
```

### 3. Encoder Presets
```rust
Encoder::new()             // Balanced (Q85, auto strategy)
Encoder::max_compression() // Low quality, trellis, progressive
Encoder::max_quality()     // High quality, adaptive quant
Encoder::fastest()         // No optimization, baseline
```

## GPU Support

### Requirements
- CUDA 12.x installed (`CUDA_PATH` environment variable)
- Build with `--features gpu`

### GPU Feature Dependencies
```toml
[features]
gpu = ["ssimulacra2-cuda", "dssim-cuda", "cudarse-driver", "cudarse-npp"]

[dependencies]
ssimulacra2-cuda = { path = "../turbo-metrics/crates/ssimulacra2-cuda", optional = true }
dssim-cuda = { path = "../turbo-metrics/crates/dssim-cuda", optional = true }
cudarse-driver = { path = "../turbo-metrics/crates/cudarse/cudarse-driver", optional = true }
cudarse-npp = { path = "../turbo-metrics/crates/cudarse/cudarse-npp", features = ["isu", "ist"], optional = true }
```

### Known GPU Issues
1. **CUDA cleanup crash**: Uses `std::process::exit(0)` at end of GPU mode
2. **Thread locality**: GPU context is thread-local, requires sequential processing
3. **Fixed dimensions**: GPU context tied to image dimensions, recreated per image

## Testing

```bash
cargo test                    # Run all tests
cargo test --release          # With optimizations
cargo test encode             # Only encoder tests
cargo test -- --nocapture     # Show output
```

## Dependencies

### Core
- `mozjpeg-oxide` (path: ../mozjpeg-rs) - mozjpeg Rust port
- `jpegli` (path: ../jpegli-rs/jpegli-rs) - jpegli Rust port
- `bytemuck` - Safe transmutes
- `wide` - SIMD

### Dev/Testing
- `codec-eval` (path: ../codec-eval) - Quality metrics and comparison
- `butteraugli` (path: ../butteraugli/butteraugli) - Perceptual quality metric
- `fast-ssim2` - Fast SSIMULACRA2 (CPU)
- `dssim` - DSSIM quality metric
- `png` - Image I/O
- `jpeg-decoder` - JPEG verification

### GPU (optional)
- `ssimulacra2-cuda` (path: ../turbo-metrics) - GPU SSIMULACRA2
- `dssim-cuda` (path: ../turbo-metrics) - GPU DSSIM
- `cudarse-driver` - CUDA driver bindings
- `cudarse-npp` - NPP image processing

## Workflow

### Making Changes
1. Run `cargo fmt` before changes
2. Make changes
3. Run `cargo test`
4. Commit with descriptive message

### Running Benchmarks
```bash
# Full benchmark with GPU acceleration
CUDA_PATH=/usr/local/cuda-12.6 cargo run --release --features gpu \
  --example discover_heuristics -- \
  --corpus ~/work/codec-corpus/kodak \
  --output ./results \
  --gpu

# CPU-only benchmark
cargo run --release --example discover_heuristics -- \
  --corpus ~/work/codec-corpus/kodak \
  --output ./results

# Quick test (1 image)
cargo run --release --example discover_heuristics -- \
  --corpus ~/work/codec-corpus/kodak \
  --output ./results \
  --max-images 1
```

## Quality Metrics

**Use DSSIM or SSIMULACRA2, NOT PSNR/MSE.**

| Metric | Package | Range | Notes |
|--------|---------|-------|-------|
| SSIMULACRA2 | fast-ssim2 | 100 = identical | Primary (GPU accelerated) |
| Butteraugli | butteraugli | <1.0 good | Perceptual |
| DSSIM | dssim | 0 = identical | Structural |

## Corpus Locations

- **Kodak**: `~/work/codec-corpus/kodak/` (24 images, 768x512 / 512x768)
- **CID22**: `~/work/codec-corpus/CID22/CID22-512/training/` (209 images, 512x512)

## References

- [mozjpeg-rs CLAUDE.md](../mozjpeg-rs/CLAUDE.md) - mozjpeg implementation details
- [jpegli-rs CLAUDE.md](../jpegli-rs/jpegli-rs/CLAUDE.md) - jpegli implementation details
- [codec-eval CLAUDE.md](../codec-eval/CLAUDE.md) - Evaluation methodology
- [glassa CLAUDE.md](../glassa/CLAUDE.md) - GPU optimization for quantization tables
