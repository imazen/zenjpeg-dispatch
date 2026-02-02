# Context Handoff: zenjpeg-dispatch

**Date**: 2026-01-23
**Branch**: `delegate-to-dependencies`
**Last Commit**: `9012d31` - Rename from zenjpeg to zenjpeg-dispatch

## What Was Done

### Rename Complete
The project was renamed from `zenjpeg` to `zenjpeg-dispatch`:
- Directory: `/home/lilith/work/zenjpeg` → `/home/lilith/work/zenjpeg-dispatch`
- Package name in Cargo.toml updated
- All doc comments updated to reflect dispatcher role

### Dependency Update
The old `jpegli` dependency (which pointed to non-existent `../jpegli-rs/jpegli-rs`) was replaced:
```toml
# Old (broken):
jpegli = { path = "../jpegli-rs/jpegli-rs", package = "jpegli-rs" }

# New (working):
zenjpeg-encoder = { path = "../jpegli-rs/zenjpeg", package = "zenjpeg" }
```

### API Migration
The `encode_rgb_with_jpegli` and `encode_gray_with_jpegli` functions in `src/encode.rs` were updated to use the new zenjpeg encoder API:

**Old API:**
```rust
jpegli::Encoder::new()
    .width(width as u32)
    .height(height as u32)
    .pixel_format(jpegli::PixelFormat::Rgb)
    .quality(jpegli::quant::Quality::Traditional(q))
    .encode(pixels)
```

**New API:**
```rust
use zenjpeg_encoder::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable};

let config = EncoderConfig::ycbcr(q, ChromaSubsampling::None);
let mut encoder = config.encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)?;
encoder.push_packed(pixels, Unstoppable)?;
encoder.finish()
```

## Current State

### Build Status
- **Library compiles** with 152 warnings (unused code, multiversion cfg warnings)
- **Examples/tests NOT updated** - they still reference `zenjpeg::` and will fail

### Files Committed (in 9012d31)
- `Cargo.toml`
- `CLAUDE.md`
- `src/lib.rs`
- `src/error.rs`
- `src/types.rs`
- `src/encode.rs`
- `src/unified_quality.rs`

### Files Still Modified (from before rename)
These were modified before the rename and remain uncommitted:
- Examples: `deringing_benchmark.rs`, `discover_heuristics.rs`, `hybrid_encoder.rs`, `pareto_benchmark.rs`, `quality_check.rs`, `sa_tables_benchmark.rs`, `timing_benchmark.rs`, `verify_decoders.rs`
- Source: `adaptive_config.rs`, `analysis.rs`, `bpp_mapping.rs`, `dct.rs`, `entropy.rs`, `huffman.rs`, `sa_tables.rs`
- Tests: `aq_locked_tests.rs`, `common/mod.rs`, `cpp_subprocess_comparison.rs`

## What Needs To Be Done

### High Priority
1. **Update examples and tests** - They use `zenjpeg::` which now refers to the local crate (zenjpeg-dispatch), but many may expect the encoder functionality that's now in `zenjpeg_encoder`

2. **Decide on crate purpose** - The new zenjpeg at `jpegli-rs/zenjpeg` already has:
   - `Quality::ApproxJpegli(f32)`
   - `Quality::ApproxMozjpeg(u8)`
   - `Quality::ApproxSsim2(f32)`
   - `Quality::ApproxButteraugli(f32)`

   This may make zenjpeg-dispatch redundant, or its role should be clarified.

3. **Clean up unused code** - Many warnings about dead code in:
   - `src/trellis.rs` (compute_block_eob_info, simple_quantize_block)
   - `src/strategy.rs` (select_strategy_for_image, strategy_from_analysis, compute_aq_strength_from_analysis)

### Low Priority
- Update multiversion crate to silence cfg warnings
- Review if mozjpeg-oxide dependency is still needed

## Key Files

| File | Purpose |
|------|---------|
| `src/encode.rs` | Main encoder with delegation to mozjpeg-oxide or zenjpeg-encoder |
| `src/strategy.rs` | Strategy selection logic (mozjpeg vs jpegli) |
| `src/analysis.rs` | Image analysis for codec selection |
| `src/unified_quality.rs` | Quality scale normalization |

## Related Repositories

- **New zenjpeg encoder**: `/home/lilith/work/jpegli-rs/zenjpeg` (the main encoder)
- **mozjpeg-rs**: `/home/lilith/work/mozjpeg-rs` (mozjpeg Rust port)
- **turbo-metrics**: `/home/lilith/work/turbo-metrics` (GPU-accelerated quality metrics)

## Commands

```bash
cd /home/lilith/work/zenjpeg-dispatch
cargo check                    # Verify build
cargo test                     # Run tests (may have failures)
git log -5                     # Recent commits
git diff --stat                # See uncommitted changes
```
